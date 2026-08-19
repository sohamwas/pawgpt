"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  askStream,
  getHealth,
  type Health,
  type StreamEvent,
  type Trace,
} from "@/lib/api";
import { Answer } from "../Answer";
import { Inspector } from "../Inspector";

type Stage = { stage: string; detail: string };

type Message = {
  role: "user" | "assistant";
  text: string;
  stages?: Stage[];
  trace?: Trace;
  error?: string;
  streaming?: boolean;
};

/* Chosen to show all three retrieval modes: the first is pure structured filtering,
   the second has no corresponding column at all, the third is a single named breed. */
const EXAMPLES = [
  {
    q: "I live in a small apartment and work 9 hours a day. I need a dog that won't bark much and can handle being alone.",
    why: "matched on barking, time alone and apartment suitability",
  },
  {
    q: "Which breeds tend to have hip dysplasia problems?",
    why: "searches the full breed write-ups",
  },
  {
    q: "I want a dog under 12kg that is good with young children and easy to train for a first-time owner.",
    why: "narrows by size and temperament, then compares",
  },
];

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth().then(setHealth);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(question: string) {
    const q = question.trim();
    if (!q || busy) return;

    setInput("");
    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", text: q },
      { role: "assistant", text: "", stages: [], streaming: true },
    ]);

    // Only the last message is ever mutated, so updates target it by index rather
    // than rebuilding the list on every token.
    const update = (fn: (m: Message) => Message) =>
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = fn(next[next.length - 1]);
        return next;
      });

    const onEvent = (event: StreamEvent) => {
      if (event.type === "progress") {
        update((m) => ({
          ...m,
          stages: [...(m.stages ?? []), { stage: event.stage, detail: event.detail }],
        }));
      } else if (event.type === "token") {
        update((m) => ({ ...m, text: m.text + event.text }));
      } else if (event.type === "done") {
        update((m) => ({ ...m, trace: event.trace, streaming: false }));
      } else if (event.type === "error") {
        update((m) => ({ ...m, error: event.message, streaming: false }));
      }
    };

    try {
      await askStream(q, onEvent);
    } catch (err) {
      update((m) => ({
        ...m,
        error:
          err instanceof Error
            ? err.message
            : "The request failed before the answer started.",
        streaming: false,
      }));
    } finally {
      update((m) => ({ ...m, streaming: false }));
      setBusy(false);
    }
  }

  // Deliberately no chunk counts or model names: those are implementation details
  // that mean nothing to the person asking, and reading them as status noise makes
  // the product feel like a demo. Breed count is the one number worth showing.
  const statusDot = !health ? "wait" : health.ok ? "up" : "down";
  const statusText = !health
    ? "connecting…"
    : health.ok
      ? `${health.breeds} breeds ready`
      : "Service unavailable, please try again shortly.";

  return (
    <>
      <div className="shell">
        <Link href="/" className="backlink">
          <span aria-hidden>←</span> Back
        </Link>

        <header className="masthead">
          <h1>🐾 PawGPT</h1>
          <span className="sub">Tell me about your home, your hours and your patience.</span>
        </header>

        <div className="status">
          <span className={`dot ${statusDot}`} />
          {statusText}
        </div>

        {messages.length === 0 && (
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button
                key={ex.q}
                className="example"
                onClick={() => send(ex.q)}
                disabled={busy}
              >
                {ex.q}
                <span className="why">{ex.why}</span>
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="role">{m.role === "user" ? "You" : "PawGPT"}</div>
            <div className="bubble">
              {m.role === "user" ? (
                m.text
              ) : (
                <>
                  {m.stages && m.stages.length > 0 && !m.text && !m.error && (
                    <div className="progress">
                      {m.stages.map((s, j) => {
                        const last = j === m.stages!.length - 1;
                        return (
                          <div
                            key={j}
                            className={`stage ${last && m.streaming ? "active" : ""}`}
                          >
                            {last && m.streaming ? (
                              <span className="spinner" />
                            ) : (
                              <span className="mark">✓</span>
                            )}
                            {s.detail}
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {m.error && <div className="banner error">{m.error}</div>}

                  {m.text && (
                    <Answer text={m.text} streaming={!!m.streaming} trace={m.trace} />
                  )}

                  {m.trace && <Inspector trace={m.trace} stages={m.stages ?? []} />}
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <div className="composer-inner">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe the dog you're looking for…"
            disabled={busy}
            aria-label="Your question"
          />
          <button type="submit" disabled={busy || !input.trim()}>
            {busy ? "…" : "Ask"}
          </button>
        </div>
      </form>
    </>
  );
}
