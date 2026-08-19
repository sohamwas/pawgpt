/**
 * Client for the PawGPT streaming API.
 *
 * The browser's built-in EventSource only issues GET requests, and the question has
 * to go in a body, so this parses the Server-Sent Events frames off a fetch stream
 * by hand. That is the whole reason this file exists.
 */

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export type ToolCall = {
  name: string;
  args: Record<string, string | number | boolean | string[]>;
  repeated: boolean;
  summary: string;
};

export type Trace = {
  calls: ToolCall[];
  modes: string[];
  breeds: string[];
  answer_context_tokens: number;
  error: string | null;
};

export type Health = {
  ok: boolean;
  breeds?: number;
  index?: { chunks: number; dim: number } | null;
  tool_model?: string;
  generation_model?: string;
  groq_key_present?: boolean;
  error?: string;
};

export type StreamEvent =
  | { type: "progress"; stage: string; detail: string }
  | { type: "token"; text: string }
  | { type: "done"; trace: Trace }
  | { type: "error"; message: string };

export async function getHealth(): Promise<Health> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
    return await res.json();
  } catch (err) {
    return { ok: false, error: `Cannot reach the API at ${API_URL}` };
  }
}

/**
 * POST a question and invoke `onEvent` as each SSE frame arrives.
 *
 * Frames are `event: <name>\ndata: <json>\n\n`. A chunk from the network can split
 * anywhere, including mid-frame, so incomplete text is held in a buffer until the
 * blank-line terminator shows up rather than parsed optimistically.
 */
export async function askStream(
  question: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!res.ok || !res.body) {
    onEvent({
      type: "error",
      message:
        res.status === 0
          ? `Cannot reach the API at ${API_URL}.`
          : `The API returned ${res.status}. Is the backend running?`,
    });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line; anything after the last one is a
    // partial frame and stays in the buffer for the next read.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let name = "";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7).trim();
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!name || !data) continue;

      try {
        onEvent({ type: name, ...JSON.parse(data) } as StreamEvent);
      } catch {
        // A malformed frame should not tear down a working stream.
      }
    }
  }
}
