"""
HTTP API in front of the retrieval agent.

Kept deliberately thin: every decision lives in `retrieval/`, which is tested
offline. This module's only jobs are transport, CORS, and turning the agent's event
stream into Server-Sent Events.

Streaming is the point of the design. A question takes 15-80 seconds end to end,
almost all of it waiting on a rate-limited free tier, so an endpoint that returns
only when finished would leave the page blank for a minute. Instead the client sees
the retrieval plan as it happens and then the answer word by word.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""
import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from retrieval import agent, passages, traits

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# The browser calling this is served from a different origin (the Next.js app), so
# CORS is required rather than optional. Set ALLOWED_ORIGINS to the deployed
# frontend URL; the default covers local development only.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",") if o.strip()
]

class Question(BaseModel):
    # An upper bound matters here beyond tidiness: the question is forwarded to a
    # rate-limited model API, so an unbounded field is a way to burn someone else's
    # token budget with a single request.
    question: str = Field(min_length=1, max_length=1000)


# --- rate limiting -----------------------------------------------------------
#
# /ask spends tokens from a shared, metered Groq allowance, so an open endpoint is a
# way for one visitor to exhaust the quota for everybody. This is a per-IP sliding
# window held in memory: adequate for a single instance, and honest about its limits
# - it resets on restart and is not shared between replicas. Anything larger wants
# Redis, but that would be infrastructure this project does not otherwise need.
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "12"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "300"))  # seconds
MAX_TRACKED_CLIENTS = 5_000   # bounded so the limiter cannot itself leak memory

_hits: dict[str, deque] = {}
_hits_lock = Lock()


def _client_key(request: Request) -> str:
    """Identify the caller, trusting a proxy header only when one is configured.

    Render and Vercel both sit behind proxies, so the socket address is the proxy's.
    X-Forwarded-For is spoofable when nothing is in front of the app, so it is only
    consulted when TRUST_PROXY is set - otherwise a header would defeat the limiter.
    """
    if os.environ.get("TRUST_PROXY", "").lower() in ("1", "true", "yes"):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limited(request: Request) -> int | None:
    """Return seconds to wait if the caller is over the limit, else None."""
    now = time.monotonic()
    key = _client_key(request)

    with _hits_lock:
        if len(_hits) > MAX_TRACKED_CLIENTS:
            _hits.clear()

        window = _hits.setdefault(key, deque())
        while window and now - window[0] > RATE_LIMIT_WINDOW:
            window.popleft()

        if len(window) >= RATE_LIMIT_REQUESTS:
            return int(RATE_LIMIT_WINDOW - (now - window[0])) + 1

        window.append(now)
        return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the embedding model and index before serving traffic.

    The first semantic search otherwise pays a ~10 second cold start while
    sentence-transformers loads, and on a fresh container that lands on the first
    real user rather than on deployment.
    """
    try:
        agent.warm()
        print(f"[OK] warm: index + encoder loaded, {len(traits.load())} breeds")
    except passages.IndexMissing as exc:
        # Not fatal: the structured filter works without the index, and failing to
        # boot would hide the fact that only the prose path is affected.
        print(f"[!] {exc}")
    yield


app = FastAPI(title="PawGPT API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Liveness plus enough state to diagnose a broken deploy without shell access."""
    try:
        breeds = len(traits.load())
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    # Deliberately not `load_index()`: that unpacks the 384-dimension float matrix,
    # ~34 MB the lexical backend otherwise never touches. A health check should not
    # be the thing that inflates the process it is reporting on.
    try:
        if passages.RETRIEVAL_BACKEND == "lexical":
            from retrieval import lexical

            texts, _, _ = lexical.load_corpus()
            index = {"chunks": len(texts), "backend": "lexical"}
        else:
            vectors, _, _, _ = passages.load_index()
            index = {"chunks": int(vectors.shape[0]),
                     "dim": int(vectors.shape[1]),
                     "backend": "dense"}
    except passages.IndexMissing:
        index = None

    return {
        "ok": True,
        "breeds": breeds,
        "index": index,
        "tool_model": agent.TOOL_MODEL,
        "generation_model": agent.GENERATION_MODEL,
        "groq_key_present": bool(os.environ.get("GROQ_API_KEY")),
    }


@app.get("/traits")
def list_traits():
    """The filterable traits and their ranges - lets the UI build controls itself."""
    out = []
    for name in traits.FILTERABLE:
        kind = traits.kind(name)
        entry = {"name": name, "kind": kind, "label": name.replace("_", " ")}
        if kind == traits.CATEGORICAL:
            entry["values"] = sorted(traits.load()[traits.column(name)].unique())
        else:
            lo, hi = traits.scale(name)
            entry["min"], entry["max"] = lo, hi
        out.append(entry)
    return {"traits": out}


def _sse(event_type, payload):
    """One Server-Sent Event. The blank line terminates the frame."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Headers that cost nothing and close off whole categories of attack.

    This API returns JSON and an event stream, never HTML, so a page has no business
    being framed, sniffed into another content type, or treated as a document origin.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response


@app.post("/ask")
async def ask(body: Question, request: Request):
    """Stream the answer to a question as Server-Sent Events.

    Events: `progress` (what retrieval is doing), `token` (answer text, incremental),
    `done` (the trace, for the inspector), `error` (a readable message).
    """
    wait = rate_limited(request)
    if wait is not None:
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Too many questions. Try again in about {wait} seconds.",
                "retry_after": wait,
            },
            headers={"Retry-After": str(wait)},
        )

    async def events():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            # The agent is synchronous and network-bound, so it runs in a worker
            # thread and hands events back through the queue. Without this the event
            # loop would block and nothing would flush until the whole answer was
            # finished - which is exactly what streaming is meant to avoid.
            try:
                for event in agent.ask_stream(body.question):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # never leak a traceback to the browser
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "message": agent._explain_api_error(exc)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_running_loop().run_in_executor(None, produce)

        while True:
            event = await queue.get()
            if event is None:
                break
            kind = event.pop("type")
            yield _sse(kind, event)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Render and most reverse proxies buffer responses by default, which
            # would defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )
