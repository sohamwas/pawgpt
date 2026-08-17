# PawGPT — Engineering Notes

A record of every problem found in this project, why each one mattered, how it was
fixed, and how the fix was verified. Written to be read cold — by a future me, or by
someone asking "so what did you actually do here?"

Each entry follows the same shape: **Symptom → Root cause → Fix → Verification**.

---

## Contents

**Retrieval correctness (the ones that changed answer quality)**
1. [Embeddings only saw 2% of each document](#1-embeddings-only-saw-2-of-each-document)
2. [The LLM never read the text the search matched on](#2-the-llm-never-read-the-text-the-search-matched-on)

**Reproducibility (a fresh clone could not be run)**
3. [The setup guide pointed at a file that did not exist](#3-the-setup-guide-pointed-at-a-file-that-did-not-exist)
4. [The data path broke when run from the repo root](#4-the-data-path-broke-when-run-from-the-repo-root)
5. [Unpinned dependencies](#5-unpinned-dependencies)
6. [PyTorch install failed on Windows](#6-pytorch-install-failed-on-windows)

**Secrets**
7. [An API key hardcoded in source](#7-an-api-key-hardcoded-in-source)
8. [Keys moved to a single .env](#8-keys-moved-to-a-single-env)

**Repo hygiene**
9. [The local folder was not the deployed project](#9-the-local-folder-was-not-the-deployed-project)
10. [Documentation drift](#10-documentation-drift)
11. [Unanchored .gitignore patterns](#11-unanchored-gitignore-patterns)

---

# Retrieval correctness

These two are the interesting ones. Everything else is hygiene; these changed what
the app actually says.

## 1. Embeddings only saw 2% of each document

### Symptom

The README already admitted it: *"Occasionally, the assistant may respond with 'Not
Enough Information'."* Answers were vague and often missed facts that were plainly
present in the dataset.

### Root cause

Some background on how RAG works: to make text searchable by meaning, you convert it
into a vector — a list of numbers — using an *embedding model*. Similar meanings land
near each other in that space. Every embedding model has a hard input limit measured
in **tokens** (roughly, word-pieces). Text past that limit is silently discarded.

This project used `all-MiniLM-L6-v2`, whose limit is **256 tokens**.

The dataset's `Combined_Info` column holds a full breed write-up — description,
temperament, exercise, grooming, health, choosing a breeder. Measured directly:

```
all-MiniLM-L6-v2 max_seq_length = 256 tokens
breed 'Afador': 64,011 chars -> 12,660 tokens
tokens actually embedded: 256 of 12,660  (2.0%)
```

The original script embedded the whole document in one call:

```python
doc_text = str(row['Combined_Info'])
vector = embeddings_model.embed_query(doc_text)   # 12,660 tokens in, 256 used
```

So each breed was represented by a single vector built from its **first ~2%** — the
opening description paragraph. Everything about health, grooming, training and
exercise was never encoded at all. The library said so, in a warning easy to miss:

```
Token indices sequence length is longer than the specified maximum
sequence length for this model (12660 > 256).
```

**Why this is the kind of bug that hides:** nothing crashes. You get a vector back.
Search returns results. Results look plausible, because the first paragraph of a dog
breed article is genuinely about that dog. The failure only shows up as a slow drip
of "I don't have enough information" on questions whose answers lived in the other 98%.

### Fix

Split each breed into chunks that fit the model, and embed each chunk separately —
standard practice in RAG, and the reason "chunking strategy" is a thing people argue
about.

Two decisions worth explaining:

**Split on real tokens, not characters.** A character-count guess (`chunk_size=1000`)
drifts, because token-to-character ratio varies with the text. Using the model's own
tokenizer makes `chunk_size` mean exactly what it says:

```python
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=220, chunk_overlap=32,
)
```

`220`, not `256`, leaves headroom for the breed-name prefix added next. The `32`-token
overlap means a sentence sitting on a chunk boundary still appears whole in one of
them — without overlap, boundary sentences become unretrievable.

**Prefix every chunk with its breed name.** A chunk from the middle of a document
reads like this:

```
"capabilities for a mutually rewarding relationship. For first-time or novice
dog owners, pet ownership can be both exciting and daunting..."
```

Nothing identifies the breed. That chunk cannot be retrieved by a query naming the
breed, and if it *were* retrieved, the LLM would have no idea which dog it describes.
So each chunk is stored as `f"Breed: {breed}. {chunk}"`.

Vector IDs changed from `"{row}"` to `"{row}-{chunk}"`, so the script now clears the
index before repopulating — otherwise the old one-vector-per-breed entries would
survive forever and keep polluting results.

### Verification

Ran the shipped config against the real tokenizer over 12 breeds / 827 chunks:

```
MODEL_MAX_TOKENS=256 CHUNK_TOKENS=220 OVERLAP=32
Afador          12,660 tok ->  68 chunks (max 227 tok/chunk)
Affenhuahua     12,149 tok ->  65 chunks (max 229 tok/chunk)
Affenpinscher   13,658 tok ->  73 chunks (max 229 tok/chunk)

[PASS] chunks exceeding 256-token model limit: 0
worst chunk: 230 tokens, headroom = 26
[PASS] mid-document chunk carries its breed name
```

**Result: coverage went from 2% of each document to 100%.** The index grows from 391
vectors to roughly 27,000.

### If asked "what would you do next?"

Use a model with a longer context (many handle 512–8192 tokens) so chunks can be
bigger and carry more context each; or add a re-ranking pass over the top ~50 chunks.
Also worth stripping the generic boilerplate — advice about choosing a breeder is
near-identical across all 391 breeds, so those chunks compete with breed-specific ones.

---

## 2. The LLM never read the text the search matched on

### Symptom

Same as above, and easy to conflate with it — but a genuinely separate bug. Even for
the 2% that *was* embedded correctly, the model's answer did not draw on it.

### Root cause

A vector database stores a vector plus arbitrary **metadata** you attach to it. You
search on the vector, then read the metadata to see what you found.

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

Worth stating plainly in an interview, because it is the more instructive of the two
bugs: the retrieval half and the generation half disagreed about what a "document"
was. Each half was internally reasonable. The seam between them was wrong.

### Fix

Store the chunk's own text in its metadata, and build the prompt from that:

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
  attribute each fact to the breed whose passage it came from. Without this an LLM will
  cheerfully blend two breeds into one confident answer.
- **A loud failure instead of a silent one.** If matches come back with no `text`
  field, the app says the index predates chunking and to re-run the script — rather
  than falling back to the old `str(metadata)` behaviour and hiding the problem.

### Verification

Static: index name, embedding model and vector dimension are consistent across both
files; every chunk carries non-empty `text` and `breed_name`. End-to-end answer quality
requires re-running the population script against live Pinecone — see
[Status](#status) below.

---

# Reproducibility

The test that matters: *can someone clone this and run it?* Before these fixes, no.

## 3. The setup guide pointed at a file that did not exist

**Symptom.** Following the README verbatim failed at the first command.

**Root cause.** README said `python scripts/populate_pinecone.py`. The file is
`scripts/pinecone_db.py`. It had been renamed at some point; the docs were not.

**Fix.** Corrected the README to the real filename.

**Verification.** An automated check now asserts that every path the README mentions
exists on disk, and that the stale names (`populate_pinecone.py`, `pawgpt-dog-breeds`,
`your-username`) appear nowhere in it.

---

## 4. The data path broke when run from the repo root

**Symptom.** `FileNotFoundError` immediately after fixing #3.

**Root cause.** The script did `pd.read_csv("dogs_final_for_rag.csv")` — a *relative*
path. The dataset lives in `data/`. The path only resolved if you happened to run the
script from a directory containing a copy of the CSV.

**Fix.** Resolve from the file's own location, so the working directory stops mattering:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
csv_path = REPO_ROOT / "data" / "dogs_final_for_rag.csv"
```

The same pattern anchors `.env` loading in all three entry points.

**Verification.** Asserted that the resolved path points at a file that exists.

---

## 5. Unpinned dependencies

**Symptom.** None yet — this is the bug that arrives months later, when a clone that
worked in July stops working in November.

**Root cause.** `requirements.txt` listed bare names:

```
streamlit
pinecone
langchain
langchain-community
```

Every install got whatever was newest that day. For a stack moving as fast as
LangChain, two people installing a week apart get materially different libraries.

**Fix.** Pinned every direct dependency to an exact version, resolved together by pip
so they are known mutually compatible:

```
streamlit==1.61.1
pinecone==9.1.0
langchain-community==0.4.2
langchain-groq==1.1.3
langchain-text-splitters==1.1.2
sentence-transformers==5.7.0
transformers==5.15.0
pandas==3.0.5
python-dotenv==1.2.3
```

Also **removed the top-level `langchain` package** — parsing the imports of both source
files showed nothing referenced it. It was inherited weight.

**Verification.** Installed the pinned set into a clean virtual environment, then
imported every module the app uses and compared installed versions against the pins:

```
[PASS] streamlit 1.61.1   [PASS] pinecone 9.1.0   [PASS] pandas 3.0.5
[PASS] langchain-community 0.4.2   [PASS] sentence-transformers 5.7.0
[PASS] from langchain_community.embeddings import HuggingFaceEmbeddings
[PASS] from langchain_groq import ChatGroq
[PASS] from pinecone import Pinecone, ServerlessSpec
```

An automated check also confirms every import maps to something in `requirements.txt`,
so a future import cannot quietly go unpinned.

---

## 6. PyTorch install failed on Windows

**Symptom.**

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file
or directory: '...\torch\include\ATen\native\transformers\cuda\
mem_eff_attention\iterators\predicated_tile_access_iterator_residual_last.h'
```

**Root cause.** Not a dependency conflict — a Windows one. Windows has a legacy
260-character limit on file paths (`MAX_PATH`), disabled-by-default long path support
(`LongPathsEnabled = 0`), and PyTorch ships CUDA headers nested absurdly deep. Add a
long virtual-environment path and you cross 260 characters mid-install.

The distinction matters: the same `requirements.txt` **failed at a 131-character venv
path and succeeded at a 36-character one**. Nothing about the dependencies was wrong.

**Fix.** Documented in the README: create the venv inside the project (`.venv`, ~36
chars), or enable long paths:

```powershell
Set-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" LongPathsEnabled 1
```

**Verification.** Reproduced the failure at the long path, then a clean `pip install
-r requirements.txt` at the short path exited 0.

**Lesson worth keeping:** read the actual error before blaming the obvious suspect. The
pinned versions were newer than the ones previously deployed, so "the new pins broke
it" was the tempting conclusion — and it was wrong. The traceback named a filesystem
error and pip printed a hint about long paths.

---

# Secrets

## 7. An API key hardcoded in source

**Symptom.** A live Groq API key sat in plaintext in two places:

```python
# app.py
GROQ_API_KEY = "gsk_0fHy..."      # a real, working key
```

plus a `groq_api_key.txt` file beside it, in a cloud-synced folder.

**Root cause.** Ordinary shortcut-taking during a first build — hardcode it, mean to
fix it later.

**Fix.** Removed both. The key now comes from the environment only, and the code
fails loudly rather than proceeding with a missing credential:

```python
if not os.environ.get("GROQ_API_KEY"):
    sys.exit("❌ GROQ_API_KEY is not set.\n"
             "   Copy .env.example to .env and fill in your Groq key.")
```

`groq_api_key.txt` was deleted; the population script's `"YOUR_API_KEY_HERE"`
placeholder (which required editing source to run) became an environment read.

**Verification.** Scanned every file in the folder — including gitignored ones — for
`gsk_`/`pcsk_` patterns. Exactly one hit: `.env`, which git cannot see. Also scanned
all 50+ commits of history; the key appears in none of them, so no history rewrite
was needed.

**The important caveat:** removing a key from a file does not un-leak it. A key that
has been sitting in plaintext should be **rotated at the provider**. Deleting the file
controls where it is stored; only rotation makes the old value worthless.

---

## 8. Keys moved to a single .env

**Symptom.** `st.secrets["groq_api_key"]` raises if no `secrets.toml` exists, so the
app was effectively Streamlit-Cloud-only and gave an opaque error anywhere else.

**Fix.** One `.env` file at the repo root, read by all three entry points:

```python
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
```

`override=False` is the load-bearing detail: **real environment variables win over the
file**. Locally you get `.env`; on Render or Vercel you set the variables in the
dashboard and ship no `.env` at all. Same code, no branching.

`.env.example` is committed as a template; `.env` is gitignored. `st.secrets` remains
as a last-resort fallback so the existing Streamlit Cloud deployment keeps working
until it is migrated.

**Verification.** Eight assertions against a synthetic `.env` fixture: the file
populates both keys; a real environment variable beats the file; a missing `.env` does
not raise (important — a fresh clone has none); and `.env.example` contains no
real-looking key.

---

# Repo hygiene

## 9. The local folder was not the deployed project

**Symptom.** The working folder had no `.git` at all, and its `streamlit_app.py` posted
to `http://localhost:5000/ask` — a Flask backend. The deployed app queries Pinecone
directly. Running the local copy standalone returned a connection error on every message.

**Root cause.** The folder was a detached snapshot of an **older architecture**
(Flask + local ChromaDB), left behind after the project moved to Streamlit + Pinecone.
The repo's history confirmed it: `app.py`, `flask_app.py`, `template/index.html` and
`db_chroma/` had all been deliberately deleted in earlier commits.

**Fix.** Converted the folder into a real clone: installed `.git`, restored the tracked
files, and removed root-level duplicates — but only after hash-comparing each against
its repo counterpart. Four matched exactly. The CSV differed by 396 bytes across 396
lines — exactly one byte per line, i.e. git's CRLF line-ending conversion, not a content
difference.

The old Flask/ChromaDB prototype was kept on disk but gitignored: still available for
reference, deliberately not part of the repo.

**Lesson:** verify before deleting. "These look like the same file" and "these hash
identically" are different claims, and the 396-byte discrepancy would have looked like
data loss without checking why.

---

## 10. Documentation drift

Small individually, collectively the difference between a repo that reads as
maintained and one that does not:

| Claim in README | Reality |
|---|---|
| `scripts/populate_pinecone.py` | file is `scripts/pinecone_db.py` |
| creates index `pawgpt-dog-breeds` | both script and app use `pawgpt` |
| `demo/pawgpt_recording.mp4` | repo has `.mov` |
| `git clone .../your-username/pawgpt.git` | placeholder never filled in |
| — | no step told you to configure API keys at all |

All corrected, and the last gap — the missing key-configuration step — was the one
that actually blocked people.

**Verification.** An automated check asserts every README-referenced path exists and
every stale string is absent, so this class of drift gets caught rather than
rediscovered.

---

## 11. Unanchored .gitignore patterns

**Symptom.** None yet — caught while reviewing my own work.

**Root cause.** The `.gitignore` I wrote listed bare filenames:

```
app.py
vector_db.py
```

A `.gitignore` pattern without a slash matches **at any depth**. So a future
`scripts/app.py` would have been silently untracked — the sort of thing discovered
much later, via a file that mysteriously never made it to production.

**Fix.** Anchored to the repo root, where the legacy prototype actually lives:

```
/app.py
/vector_db.py
/templates/
/db_chroma/
```

**Verification.** Confirmed both directions: the root-level legacy files are still
ignored, and a test file at `scripts/tmpcheck/app.py` is correctly visible to git.

---

# Status

**Done and verified:** everything above except the end-to-end retrieval-quality check.

**Still required:** re-run `python scripts/pinecone_db.py` to rebuild the index with
chunked vectors. Until that runs, the deployed app still queries the old
one-vector-per-breed index, and fixes #1 and #2 are not live. The app will say so
explicitly rather than degrading quietly.

**Also outstanding:** rotate the exposed Groq key; merge and push this branch
(`origin/main` still has the old code); and migrate off the deprecated
`langchain_community.embeddings.HuggingFaceEmbeddings` before the next major upgrade.

---

# Summary

| # | Problem | Impact | Fixed |
|---|---|---|---|
| 1 | Embeddings truncated at 256 of ~12,660 tokens | 98% of each document unsearchable | ✅ chunking |
| 2 | Prompt built from `str(metadata)` | LLM never read the retrieved prose | ✅ chunk text in metadata |
| 3 | README named a non-existent script | setup failed at step one | ✅ |
| 4 | Relative CSV path | `FileNotFoundError` from repo root | ✅ |
| 5 | Unpinned dependencies | installs drift over time | ✅ 9 exact pins |
| 6 | PyTorch vs Windows `MAX_PATH` | install fails on Windows | ✅ documented |
| 7 | API key hardcoded in source | live credential in plaintext | ✅ (rotation still needed) |
| 8 | `st.secrets` only | app was Streamlit-Cloud-only | ✅ `.env` |
| 9 | Folder was not the deployed code | edits would not reach production | ✅ real clone |
| 10 | Documentation drift | README described a different repo | ✅ |
| 11 | Unanchored ignore patterns | future files silently untracked | ✅ |

The two-line version, if someone asks what this project taught me: **a RAG pipeline can
be completely broken while every component reports success** — the embedder returned
vectors, the database returned matches, the LLM returned fluent answers, and 98% of the
data was never in play. And **a pinned, documented, secret-free setup is not
bureaucracy** — it is the difference between a project someone else can run and a
project that only ever worked on one laptop.
