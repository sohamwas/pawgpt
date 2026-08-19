# Deploying PawGPT

Two services, two hosts. The frontend is a Next.js app on Vercel; the API is a
Python service on Render.

**Both halves run on free tiers.** Getting there needed one real change: the
deployed API uses BM25 rather than embeddings, because `sentence-transformers` pulls
in torch, and torch is 395 MB of what was a 615 MB service against a 512 MB free
ceiling. Without it the service measures **145 MB**. See "Why BM25 in production"
below for what that costs.

They are two hosts rather than one because the Python service cannot run on Vercel
regardless: a question takes 10 to 30 seconds, past the serverless execution window,
and it is a long-lived process holding an in-memory index rather than a function.

```
┌──────────────────────┐   HTTPS + SSE   ┌────────────────────────────┐
│ Vercel  (free)       │ ──────────────► │ Render  (free)             │
│ Next.js  (web/)      │ ◄────────────── │ FastAPI  (api/, retrieval/)│
│ static, CDN          │                 │ 145 MB RSS, cap is 512 MB  │
└──────────────────────┘                 └────────────────────────────┘
```

**Deploy the API first.** The frontend needs its URL at build time.

---

## 0. Before you start

Commit everything, including the two generated retrieval artifacts.

```bash
git add -A
git status          # confirm .env and INTERVIEW_NOTES.md are NOT listed
git commit -m "Add agentic retrieval, API and web frontend"
git push
```

`data/breed_index.npz` (7.9 MB) and `data/breed_prose.json` (4.0 MB) are committed
on purpose. Regenerating them means embedding 4,801 chunks on CPU, which took over
ten minutes locally. Paying that inside a build timeout on every deploy, for a
byte-identical result, is worse than carrying 12 MB in the repo.

Check `.env` is not in the commit. It holds your Groq key and is gitignored; keys go
in the two dashboards instead.

---

## 1. The API, on Render

1. Go to **render.com** → **New** → **Web Service**, and connect the repository.
2. Render reads `render.yaml` and fills most of this in. Confirm:

   | Setting | Value |
   |---|---|
   | Root directory | *(blank, the repo root)* |
   | Runtime | Python 3 |
   | Build command | `pip install -r requirements-api.txt` |
   | Start command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
   | Health check path | `/health` |
   | Instance type | **Free** |

3. Add environment variables:

   | Key | Value |
   |---|---|
   | `GROQ_API_KEY` | your key from console.groq.com |
   | `TRUST_PROXY` | `true` |
   | `RETRIEVAL_BACKEND` | `lexical` |
   | `ALLOWED_ORIGINS` | leave blank for now, filled in at step 3 |

4. Deploy. The slim requirements install in about a minute.

**Install `requirements-api.txt`, not `requirements.txt`.** That is the whole
free-tier story. The slim file omits `sentence-transformers`, so torch never lands
on the instance: measured **145 MB** resident against the free cap of 512 MB, with
367 MB of headroom. Installing the full file instead takes the service to 615 MB and
it will be OOM-killed on boot.

**`TRUST_PROXY` matters more than it looks.** Render terminates TLS at a proxy, so
every request reaches the app from the proxy's address. Without this the rate
limiter sees all your visitors as one client, and a single person's twelve questions
locks out everybody else. With it set, the limiter reads `X-Forwarded-For`.

When it is live, check it:

```bash
curl https://<your-service>.onrender.com/health
```

You want `"ok": true`, `"breeds": 391`,
`"index": {"chunks": 4801, "backend": "lexical"}` and `"groq_key_present": true`.

If `index` is `null`, the artifacts did not get committed; the structured filter
still works but prose search is disabled. If `backend` says `dense`, the service is
trying to load torch and will be killed.

---

## 2. The frontend, on Vercel

1. Go to **vercel.com** → **Add New** → **Project**, and import the same repository.
2. Set **Root Directory** to `web`. This is the step people miss. Everything else is
   detected automatically.

   | Setting | Value |
   |---|---|
   | Framework preset | Next.js |
   | Root directory | **`web`** |
   | Build command | *(default)* |
   | Install command | *(default)* |

3. Add one environment variable:

   | Key | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://<your-service>.onrender.com` |

   No trailing slash.

4. Deploy.

**`NEXT_PUBLIC_` variables are compiled into the JavaScript bundle, not read at
runtime.** Changing this value in the dashboard does nothing on its own; you have to
redeploy for it to take effect. If the site loads but every question fails, this is
the first thing to check.

---

## 3. Point them at each other

Back in Render, set `ALLOWED_ORIGINS` to your Vercel URL and save. The service
restarts automatically.

```
ALLOWED_ORIGINS=https://pawgpt.vercel.app
```

Exact match: correct scheme, no trailing slash, no path. The browser compares the
`Origin` header character for character, and a trailing slash is a mismatch.

Multiple origins are comma-separated:

```
ALLOWED_ORIGINS=https://pawgpt.vercel.app,https://pawgpt-yourname.vercel.app
```

**Preview deployments will fail CORS.** Vercel gives every branch and pull request a
unique hostname, and those are not in the list. Either add the ones you use, or
accept that only production talks to the API. Do not switch to `allow_origins=["*"]`
to make this go away: it opens your Groq quota to any page on the internet.

---

## 4. Check it end to end

Open the Vercel URL. The chat page should show **391 breeds ready** rather than
"connecting…" or "Service unavailable". Then ask something real, such as
*"I live in a small apartment and work 9 hours a day and need a dog that won't bark
much"*, and confirm the retrieval stages appear before the answer streams in.

If the status dot stays grey or red:

| Symptom | Cause |
|---|---|
| "Service unavailable" | `NEXT_PUBLIC_API_URL` wrong, or set after the build. Redeploy Vercel |
| Browser console shows a CORS error | `ALLOWED_ORIGINS` does not match the Vercel URL exactly |
| First request very slow, then fine | Render cold start plus the model load. Expected |
| `"index": null` in /health | Artifacts not committed. Commit `data/breed_*.{npz,json}` |
| 429 after a few questions | Working as intended, see below |

---

## 5. What will bite you in production

**Groq's free tier is shared across all your visitors.** The measured ceiling on
this account is **8,000 tokens per minute**, and one question costs a few thousand.
That is roughly one or two people at a time before requests start queueing, and it
is why answers slow from about 4 seconds to 30 or more under load. The app degrades
honestly rather than crashing, showing "wait a moment and ask again", but a real
launch wants a paid Groq tier. Nothing in the code changes; the limit simply rises.

**The rate limiter is per-instance and in memory.** It resets on restart and is not
shared between replicas, so if you ever scale beyond one instance the effective
limit multiplies. Defaults are 12 requests per 5 minutes per IP, adjustable with
`RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW`.

**Render's free instance sleeps after 15 minutes of inactivity.** The next visitor
waits roughly 30 to 60 seconds for it to wake, then gets normal speed. This is the
real cost of the free tier and there is no way around it short of paying, or pinging
the service on a schedule. `/health` is cheap and suitable for that if you want to.

---

## 6. Why BM25 in production, and what it costs

The deployed API searches the prose with BM25 rather than embeddings. That is a
deliberate trade, not a simplification, and it is worth being able to defend.

Measured over twelve topic-lookup queries, scored on whether the retrieved passage
actually discusses the thing asked about:

| | Literal topic lookup | Paraphrased wording | Latency |
|---|---|---|---|
| Dense (MiniLM, torch) | 0.906 | **0.547** | 11.9 ms |
| BM25 (no torch) | **1.000** | 0.438 | **1.1 ms** |

BM25 is better where the user's word is the corpus's word, and worse on paraphrase,
which is exactly what you would expect. *"Fur everywhere on the sofa"* scored dense
5 and BM25 0.

The reason that is acceptable here is specific to this system rather than a general
claim about BM25. Look at what the paraphrase failures are: fur on the sofa is
*shedding*, panicking when alone is *tolerates being alone*. Both are columns, and
both are now answered exactly by `filter_breeds` before prose search is ever
consulted. What still reaches the prose is the residue the columns cannot express -
hip dysplasia, hypoallergenic coats, getting on with cats - and those are lexical
lookups, BM25's strongest case.

In other words the structured filter absorbed most of the queries that made dense
retrieval worth its 395 MB.

**To switch back**, install `requirements.txt` instead, set
`RETRIEVAL_BACKEND=dense`, and move to an instance with at least 1 GB. Nothing else
changes: both retrievers return the same shape and everything above them is
unaffected.

**A caveat on that table.** Relevance is scored by whether the passage contains the
expected term, which structurally favours BM25 on the first column. Treat the margin
there as unproven and the paraphrase gap as the real finding. Settling it properly
needs the evaluation harness that does not exist yet.

**A limitation that affects both.** "Which breeds are prone to hip dysplasia" has no
small correct answer: **159 of the 391 breeds mention it**. Both retrievers return
passages that discuss dysplasia, but neither can rank breeds by how *prone* they
are, because the corpus does not say. This is the same shape as the tie problem in
the structured filter, where 51 breeds tie at minimum shedding.

**Model availability moves.** `llama-3.1-8b-instant` was retired from Groq's
catalogue mid-project and started returning 404. If answers suddenly fail with a
model error, list what your account can actually reach:

```python
from groq import Groq
print([m.id for m in Groq(api_key="...").models.list().data])
```

Then update `TOOL_MODEL` and `GENERATION_MODEL` in `retrieval/agent.py`.

---

## Environment variables, in one place

**Render (API)**

| Key | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | yes | from console.groq.com |
| `ALLOWED_ORIGINS` | yes | exact Vercel URL, comma-separated for several |
| `TRUST_PROXY` | yes | `true`, or the rate limiter treats everyone as one client |
| `RATE_LIMIT_REQUESTS` | no | default 12 |
| `RATE_LIMIT_WINDOW` | no | default 300 seconds |

**Vercel (frontend)**

| Key | Required | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | yes | Render URL, no trailing slash, needs a redeploy to change |
