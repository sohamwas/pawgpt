"use client";

import type { Trace } from "@/lib/api";

/**
 * Shows exactly how an answer was retrieved: which tools ran, with which arguments,
 * which retrieval mode fired, and how much context the writer saw.
 *
 * This is the part that makes the architecture visible. Without it the app looks
 * like any other chatbot; with it you can see that "quiet dog for a flat" became
 * three numeric constraints against 391 rows, and that a question about hip
 * dysplasia fell through to prose search because no such column exists.
 */

const MODE_LABELS: Record<string, string> = {
  filter: "structured filter over 30 measured columns",
  full_prose: "whole breed profile, no search needed",
  scoped_search: "semantic search, restricted to the filtered breeds",
  global_search: "semantic search across all 391 breeds",
};

export function Inspector({
  trace,
  stages,
}: {
  trace: Trace;
  stages: { stage: string; detail: string }[];
}) {
  const toolCount = trace.calls.length;

  return (
    <details className="inspector">
      <summary>
        retrieval trace, {toolCount} tool call{toolCount === 1 ? "" : "s"}
        {trace.modes.length > 0 && ` · ${trace.modes.join(" → ")}`}
      </summary>

      <div className="inspector-body">
        {trace.calls.map((call, i) => (
          <div className="call" key={i}>
            <span className="name">{call.name}</span>
            {"("}
            {Object.entries(call.args)
              .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
              .join(", ")}
            {")"}
            {call.repeated && ", repeated call, served from cache"}
            <div className="out">{call.summary}</div>
          </div>
        ))}

        {trace.modes.length > 0 && (
          <div className="call">
            <div>retrieval modes used</div>
            <div className="out">
              {trace.modes.map((m, i) => (
                <div key={i}>
                  {m}, {MODE_LABELS[m] ?? "unknown mode"}
                </div>
              ))}
            </div>
          </div>
        )}

        {trace.breeds.length > 0 && (
          <div className="call">
            <div>candidate breeds reaching the model ({trace.breeds.length})</div>
            <div style={{ marginTop: 6 }}>
              {trace.breeds.map((b) => (
                <span className="pill" key={b}>
                  {b}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="call">
          <div className="out">
            writer context ≈ {trace.answer_context_tokens.toLocaleString()} tokens
            {stages.length > 0 && ` · ${stages.length} stages`}
            {"\n"}
            every rating above came from the dataset, not the model
          </div>
        </div>
      </div>
    </details>
  );
}
