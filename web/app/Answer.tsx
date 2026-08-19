"use client";

import type { Trace } from "@/lib/api";

/**
 * Renders the streamed answer.
 *
 * The model replies in light markdown - bold, bullets, and tables of trait ratings -
 * so this converts just those, rather than pulling in a markdown library and its
 * sanitiser for four constructs. Nothing here interprets raw HTML: every value is
 * placed as a text node, so model output cannot inject markup.
 */
export function Answer({
  text,
  streaming,
  trace,
}: {
  text: string;
  streaming: boolean;
  trace?: Trace;
}) {
  const relaxed = trace
    ? trace.calls.some((c) => c.summary.includes("loosened"))
    : false;

  return (
    <>
      {relaxed && (
        <div className="banner warn">
          No breed met every requirement, the filter loosened one and returned the
          closest matches. The exact change is in the retrieval trace below.
        </div>
      )}
      {renderBlocks(text)}
      {streaming && <span className="caret" aria-hidden />}
    </>
  );
}

/**
 * Punctuation the model reaches for that this site does not use.
 *
 * The gpt-oss models are fond of em dashes and of non-breaking hyphens inside
 * hyphenated words ("bark‑less"), the latter of which renders as a stray glyph in
 * some fonts. Normalising at display time rather than by prompting is deliberate:
 * instructions about punctuation are followed unreliably and cost tokens on every
 * request, whereas this is exact and free.
 */
function normalisePunctuation(text: string) {
  return text
    .replace(/\s*[—]\s*/g, ", ")   // em dash
    .replace(/\s*[–]\s*/g, " - ")  // en dash
    .replace(/[‑‐]/g, "-")    // non-breaking / unicode hyphen
    .replace(/[  ]/g, " ")    // narrow and regular non-breaking space
    .replace(/ ,/g, ",")
    .replace(/,\s*,/g, ",");
}

/** Split into paragraphs, lists and tables, then render each. */
function renderBlocks(raw: string) {
  const text = normalisePunctuation(raw);
  const lines = text.split("\n");
  const blocks: React.ReactNode[] = [];
  let paragraph: string[] = [];
  let list: string[] = [];
  let table: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(<p key={blocks.length}>{inline(paragraph.join(" "))}</p>);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list.length) {
      blocks.push(
        <ul key={blocks.length}>
          {list.map((item, i) => (
            <li key={i}>{inline(item)}</li>
          ))}
        </ul>,
      );
      list = [];
    }
  };
  const flushTable = () => {
    if (table.length) {
      blocks.push(<Table key={blocks.length} rows={table} />);
      table = [];
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (line.trim().startsWith("|")) {
      flushParagraph();
      flushList();
      table.push(line.trim());
      continue;
    }
    flushTable();

    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      list.push(bullet[1]);
      continue;
    }
    flushList();

    if (!line.trim()) flushParagraph();
    else paragraph.push(line.trim());
  }

  flushParagraph();
  flushList();
  flushTable();
  return blocks;
}

function Table({ rows }: { rows: string[] }) {
  const cells = rows
    .map((r) =>
      r
        .replace(/^\||\|$/g, "")
        .split("|")
        .map((c) => c.trim()),
    )
    // The |---|---| separator row carries no data.
    .filter((r) => !r.every((c) => /^:?-{2,}:?$/.test(c)));

  if (!cells.length) return null;
  const [head, ...body] = cells;

  return (
    <div className="scroll-x">
      <table>
        <thead>
          <tr>
            {head.map((c, i) => (
              <th key={i}>{inline(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, i) => (
            <tr key={i}>
              {row.map((c, j) => (
                <td key={j}>{inline(c)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** **bold** and `code`, as React elements rather than injected HTML. */
function inline(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) parts.push(text.slice(cursor, match.index));
    const token = match[0];
    if (token.startsWith("**")) {
      parts.push(<strong key={parts.length}>{token.slice(2, -2)}</strong>);
    } else {
      parts.push(<code key={parts.length}>{token.slice(1, -1)}</code>);
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}
