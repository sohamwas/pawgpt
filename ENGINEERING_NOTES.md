# PawGPT — Engineering Notes

Two bugs in the retrieval pipeline that, together, meant the app searched one body
of text and answered from a different, much smaller one. Both are the kind that
never raise an error — every component reported success while 98% of the dataset
was never in play.

Written to be read cold, with enough background that the RAG-specific parts make
sense without prior context.

---

## Contents

1. [Embeddings only saw 2% of each document](#1-embeddings-only-saw-2-of-each-document)
2. [The LLM never read the text the search matched on](#2-the-llm-never-read-the-text-the-search-matched-on)
3. [Status](#status)

---

## 1. Embeddings only saw 2% of each document

### Symptom

The app's own README already admitted it: *"Occasionally, the assistant may respond
with 'Not Enough Information'."* Answers were vague and routinely missed facts that
were plainly present in the dataset.

### Background

To make text searchable by meaning, you convert it into a **vector** — a list of
numbers — using an *embedding model*. Similar meanings land near each other in that
space, so "quiet dog for a flat" can match a passage that never uses those words.

Every embedding model has a hard input limit measured in **tokens** (roughly,
word-pieces). Text beyond that limit is silently discarded — not rejected, not
warned about at the API level, just dropped.

This project used `all-MiniLM-L6-v2`, whose limit is **256 tokens**.

### Root cause

The dataset's `Combined_Info` column holds a full breed write-up: description,
temperament, exercise needs, grooming, health screening, choosing a breeder.
Measured directly against the real tokenizer:

```
all-MiniLM-L6-v2 max_seq_length = 256 tokens
breed 'Afador': 64,011 chars -> 12,660 tokens
tokens actually embedded: 256 of 12,660  (2.0%)
```

The population script embedded each entire document in a single call:

```python
doc_text = str(row['Combined_Info'])
vector = embeddings_model.embed_query(doc_text)   # 12,660 tokens in, 256 used
```

So every breed was represented by **one vector built from its first ~2%** — the
opening description paragraph. Everything about health, grooming, training and
exercise was never encoded at all. The library did say so, in a warning easy to
scroll past:

```
Token indices sequence length is longer than the specified maximum
sequence length for this model (12660 > 256).
```

**Why this class of bug hides so well:** nothing crashes. You get a vector back.
Search returns results. The results look plausible, because the first paragraph of
a dog-breed article genuinely is about that dog. The failure surfaces only as a slow
drip of "I don't have enough information" on questions whose answers lived in the
other 98%.

### Fix

Split each breed into chunks that fit the model, and embed each chunk separately.
This is standard RAG practice and the reason "chunking strategy" is something people
argue about.

Two decisions worth being able to defend:

**Split on real tokens, not characters.** A character-count guess (`chunk_size=1000`)
drifts, because the token-to-character ratio varies with the text — dense technical
prose tokenizes very differently from simple sentences. Using the model's own
tokenizer makes `chunk_size` mean exactly what it says:

```python
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=220, chunk_overlap=32,
)
```

`220` rather than `256` leaves headroom for the breed-name prefix added below. The
`32`-token **overlap** means a sentence sitting on a chunk boundary still appears
whole in one of them — without overlap, boundary sentences become unretrievable.

**Prefix every chunk with its breed name.** A chunk from the middle of a document
reads like this:

```
"capabilities for a mutually rewarding relationship. For first-time or novice
dog owners, pet ownership can be both exciting and daunting..."
```

Nothing in it identifies the breed. Such a chunk cannot be retrieved by a query
naming the breed, and if it *were* retrieved, the LLM would have no idea which dog it
describes. So each chunk is stored as `f"Breed: {breed}. {chunk}"`, which fixes both
problems at once.

Vector IDs changed from `"{row}"` to `"{row}-{chunk}"`, so the script now clears the
index before repopulating — otherwise the old one-vector-per-breed entries would
survive indefinitely and keep polluting results.

### Verification

Ran the shipped configuration against the real tokenizer over 12 breeds / 827 chunks:

```
MODEL_MAX_TOKENS=256 CHUNK_TOKENS=220 OVERLAP=32
Afador          12,660 tok ->  68 chunks (max 227 tok/chunk)
Affenhuahua     12,149 tok ->  65 chunks (max 229 tok/chunk)
Affenpinscher   13,658 tok ->  73 chunks (max 229 tok/chunk)

[PASS] chunks exceeding 256-token model limit: 0
worst chunk: 230 tokens, headroom = 26
[PASS] mid-document chunk carries its breed name
```

**Coverage went from 2% of each document to 100%.** The index grows from 391 vectors
to roughly 27,000.

### If asked "what would you do next?"

Move to an embedding model with a longer context window — many handle 512 to 8192
tokens — so each chunk carries more surrounding context and there are far fewer of
them. Add a re-ranking pass over the top ~50 chunks before handing them to the LLM.
And strip the generic boilerplate: advice about choosing a breeder is near-identical
across all 391 breeds, so those chunks compete with genuinely breed-specific ones.

---

## 2. The LLM never read the text the search matched on

### Symptom

Outwardly the same as #1, and easy to mistake for it — but a genuinely separate bug.
Even for the 2% that *was* embedded correctly, the model's answer did not draw on it.

### Background

A vector database stores a vector plus arbitrary **metadata** attached to it. You
search on the vector, then read the metadata to find out what you actually retrieved.
What you choose to put in metadata determines what the LLM can see.

### Root cause

The population script attached every column *except* the breed description:

```python
metadata = row.to_dict()
metadata.pop('Combined_Info', None)     # the description is dropped here
```

And the app then built the LLM's prompt out of that metadata dict — stringified:

```python
doc_text = str(metadata)                # "{'Breed Name': 'Afador', 'Dog Size': ...}"
combined_context += doc_text + "\n\n"
```

So the pipeline **searched against the description and answered from the attribute
table**. The LLM received roughly 1.2 KB of Python `dict` repr — braces, quotes and
column names — and never saw a sentence of prose.

This is the more instructive of the two bugs, and worth stating plainly: the
retrieval half and the generation half disagreed about what a "document" was. Each
half was internally reasonable and neither was obviously broken in isolation. The
seam between them was wrong.

### Fix

Store each chunk's own text in its metadata, and build the prompt from that:

```python
"metadata": {
    "breed_name": breed,
    "chunk_index": chunk_idx,
    "text": text,          # the exact passage this vector was built from
    ...
}
```

```python
chunk_text = (metadata.get('text') or "").strip()
breed = metadata.get('breed_name', 'Unknown breed')
block = f"[{breed}]\n{chunk_text}"
```

Three supporting changes:

- **Context budget.** Chunks are small now, so `top_k` went 5 → 8 and the assembled
  context is capped at 6,000 characters rather than growing unbounded.
- **Breed labelling in the prompt.** Retrieved chunks can come from several breeds at
  once, so each block is labelled `[Breed Name]` and the prompt instructs the model to
  attribute each fact only to the breed whose passage it came from. Without this, an
  LLM will cheerfully blend two breeds into one confident answer.
- **A loud failure instead of a silent one.** If matches come back with no `text`
  field, the app reports that the index predates chunking and says to re-run the
  population script — rather than falling back to the old behaviour and hiding the
  problem again.

### Verification

Static checks confirm the index name, embedding model and vector dimension are
consistent across both files, that the script stores chunk text in metadata, and that
the app reads `metadata['text']` rather than stringifying the dict. End-to-end answer
quality requires rebuilding the index — see Status.

---

## Status

**Fixed and verified in code.** Both bugs are corrected and the chunking configuration
is confirmed against the real tokenizer.

**Not yet live.** The fixes take effect only after re-running:

```bash
python scripts/pinecone_db.py
```

Until then the deployed app still queries the old one-vector-per-breed index. It will
say so explicitly rather than degrading quietly.

---

## Summary

| # | Problem | Impact | Fix |
|---|---|---|---|
| 1 | Embeddings truncated at 256 of ~12,660 tokens | 98% of each document unsearchable | Token-aware chunking at 220 tokens with 32-token overlap, breed-name prefix |
| 2 | Prompt built from `str(metadata)` | LLM never read the retrieved prose | Chunk text stored in metadata and used to build context, labelled by breed |

The one-line version, if someone asks what this project taught me: **a RAG pipeline
can be completely broken while every component reports success** — the embedder
returned vectors, the database returned matches, the LLM returned fluent answers, and
98% of the data was never in play.
