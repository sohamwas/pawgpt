# 🐾 PawGPT

**Find the dog that fits your actual life.**

Ask in plain English, get breeds that genuinely match, with the numbers behind every
recommendation.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Next.js](https://img.shields.io/badge/frontend-Next.js%2015-black.svg)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009485.svg)
![Groq](https://img.shields.io/badge/LLM-Groq-f55036.svg)
![Tests](https://img.shields.io/badge/tests-95%20offline-brightgreen.svg)

---

## The problem this solves

PawGPT began as a textbook RAG pipeline: embed the question, take the eight nearest
text chunks, generate an answer. It had a flaw that no amount of prompt tuning fixes.

The dataset has 41 columns. **About 30 of them are structured** — measured 1–5
ratings for shedding, barking, apartment suitability, tolerance of being alone, and
two dozen more. Only one column is prose.

But the questions people actually ask look like this:

> *"I live in a small flat and work nine hours a day. I need a dog that won't bark
> much and can cope with being alone."*

That is three numeric constraints and a sort. **Semantic similarity cannot express
it.** Vector search has no way to do "at most 2", to rank, to count, or to say
"nothing matches". The old pipeline discarded 27 of the 30 structured columns at
ingest and did fuzzy text matching on the rest.

## What it does instead

The model chooses how to retrieve, and can retrieve more than once.

| Tool | Used for | Example |
|---|---|---|
| `filter_breeds` | measurable requirements | quiet, small, tolerates being alone |
| `get_breed_profile` | a named breed, or survivors of a filter | "tell me about the Basenji" |
| `search_breed_text` | topics with no column at all | hip dysplasia, hypoallergenic, cats |

A real trace:

```
"small flat, out nine hours, mustn't bark much"
  → filter_breeds(max_barking=2, min_alone=4, min_apartment=4)
      1 exact match: Basenji
      2 near matches: Azawakh (misses alone), Chow Chow (misses apartment)
  → get_breed_profile([Basenji, Azawakh, Chow Chow])
  → answer, with every rating quoted from the dataset
```

Three constraints narrowed 391 breeds to the one famously barkless breed. No
embedding could have done that.

## Why the answers are trustworthy

- **Numbers come from pandas, never the model.** The dataframe produces the
  candidate set and every figure quoted. The model writes prose about a set it was
  handed, so a hallucinated rating is structurally impossible rather than unlikely.
- **It admits when nothing fits.** If no breed meets every requirement, it loosens
  the most selective one, says which, and shows the near misses. No silent
  "no results found".
- **Everything is inspectable.** Open the retrieval trace under any answer to see
  the tools called, the arguments used, and the passages retrieved.

## What was fixed along the way

Each of these was found by measuring, and the numbers are real:

| | |
|---|---|
| **82% of the corpus was boilerplate** | The same breeder advice repeated across all 391 breeds, crowding out real content. Removing it cut 25,134 chunks to **4,801** and lifted retrieval scores from 0.531 to **0.759** |
| **Embeddings saw 2% of each document** | MiniLM truncates at 256 tokens; documents were ~12,660. Fixed with token-aware chunking |
| **The model never read what search matched** | The prompt was built from a stringified metadata dict, not the retrieved prose |
| **A single question exceeded the rate limit** | ~9,200 tokens against an 8,000/min ceiling, which is why answers crawled. Split across two models, since the limit is metered per model |
| **The writer returned empty answers** | A reasoning model spending its whole output budget thinking. Diagnosed to one parameter |
| **The deployed model had been retired** | `llama-3.1-8b-instant` now 404s on Groq |

## Stack

**Next.js 15** frontend on Vercel · **FastAPI** on Render · **Groq** for inference
(`gpt-oss-120b` chooses tools, `gpt-oss-20b` writes) · **pandas** for structured
filtering · **BM25** for prose search.

Both halves run on free tiers. The deployed service holds **145 MB** resident
because it uses BM25 rather than embeddings — sentence-transformers would drag in
torch at 395 MB and blow a 512 MB cap. The trade is measured, not assumed.

## Running it

```bash
# 1. Backend
pip install -r requirements.txt
echo GROQ_API_KEY=your-key > .env  # from console.groq.com
python scripts/dedup_prose.py     # strip boilerplate
python scripts/build_index.py     # build the index (a few minutes, CPU)
uvicorn api.main:app --port 8000

# 2. Frontend, in another terminal
cd web && npm install
cp .env.local.example .env.local
npm run dev                       # http://localhost:3000
```

```bash
pytest -q                         # 95 tests, no API key, no network
```

## Known limits

- **No quality evaluation yet.** Routing correctness is measured over a handful of
  questions. That is a smoke test, not a benchmark, and it is the next thing to
  build — the structured columns make ground truth computable without labelling.
- **Answers take 10–45 seconds** on Groq's free tier, mostly queueing. A paid tier
  removes it with no code change.
- **Render's free instance sleeps** after 15 minutes, so the first visitor after a
  quiet spell waits for a cold start.
- **Ratings are subjective judgements** from a single scraped source, presented as
  integers. They are a starting point, not veterinary advice.

## License

MIT. See [LICENSE](LICENSE).
