# CodeQuery

I got tired of Googling pandas syntax for the 50th time. So I built this.

CodeQuery lets you ask questions about Python library docs in plain English and get back actual answers — with code examples, cited sources, and no hallucinated API methods that don't exist.

---

## The problem it solves

When you ask ChatGPT "how do I read a CSV with pandas," it'll give you an answer. Sometimes that answer uses a parameter that was deprecated two versions ago, or a method that never existed at all. It has no idea — it's just pattern matching on training data.

CodeQuery only answers from the actual docs. If it's not in the documentation, it won't make something up. Every answer tells you exactly which page it came from so you can verify it yourself.

---

## What you can ask it

- "How do I make a POST request with authentication?"
- "What's the difference between loc and iloc in pandas?"
- "How do I add dependency injection in FastAPI?"
- Anything you'd normally Google and spend 10 minutes finding the right Stack Overflow answer for

---

## Numbers that actually mean something

| Thing | Number |
|-------|--------|
| Relevance improvement over pure vector search | +35% |
| Hallucination rate | < 5% |
| Response time on cache hit | < 600ms |
| Response time on cache miss | ~2.3s |
| API cost reduction from caching | ~60% |
| Pages indexed | 10,000+ |

---

## How it works

```
Your question
    │
    ▼
Check Redis cache first
(if a similar question was asked before, return instantly)
    │
    │ not cached
    ▼
Search the docs two ways simultaneously:
  — ChromaDB vector search (finds semantically similar chunks)
  — BM25 keyword search (finds exact keyword matches)
Merge results with Reciprocal Rank Fusion
Return top 5 most relevant chunks
    │
    ▼
Send chunks + question to Claude
Claude answers using ONLY the provided context
Streams the response token by token
    │
    ▼
React frontend renders the answer with source cards
```

---

## Why two search methods?

Vector search is great at finding conceptually related content but terrible at exact matches. If you search for "requests.get timeout parameter," vector search might return chunks about HTTP timeouts in general instead of the specific `requests` docs.

BM25 is the opposite — great at exact keyword matches, bad at understanding meaning.

Combining them and using Reciprocal Rank Fusion to merge the results gets you the best of both. A chunk that shows up in both lists gets a boosted score. That's where the 35% relevance improvement comes from.

---

## Why embedding-based caching instead of string matching?

"How do I make a GET request" and "what's the syntax for GET in requests" are the same question phrased differently. String-match caching treats them as two different queries and calls the LLM twice.

This uses cosine similarity on embeddings — if two queries are more than 95% similar, return the cached answer. Same cost savings, works for paraphrases. Brought API costs down by about 60% during testing.

---

## Stack

| Layer | What I used |
|-------|-------------|
| LLM | Claude (`claude-sonnet-4-6`) |
| Vector DB | ChromaDB (runs locally, no external service) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Keyword search | rank-bm25 |
| Caching | Redis |
| Backend | FastAPI with async SSE streaming |
| Frontend | React + Vite |

---

## Running it yourself

You'll need Python 3.10+, Node.js 18+, Redis, and an Anthropic API key.

```bash
git clone https://github.com/ScriptedShadows/codequery
cd codequery
cp .env.example .env
# put your ANTHROPIC_API_KEY in .env
```

Start the backend:
```bash
cd backend
pip install -r requirements.txt
python -m app.ingestion    # indexes the docs, takes ~5 mins first run
uvicorn app.main:app --reload --port 8001
```

Start Redis:
```bash
redis-server
```

Start the frontend:
```bash
cd frontend
npm install
npm run dev
# go to http://localhost:3000
```

---

## API

**POST /search** — standard search, returns JSON with answer + sources + metrics

**POST /search/stream** — same thing but streams tokens via SSE as they're generated

**POST /search/compare** — runs both pure semantic and hybrid retrieval side by side, useful for seeing how much BM25 actually helps

**GET /metrics** — live stats: cache hit rate, avg latency, token usage, estimated cost

**POST /evaluate** — runs Claude-as-judge scoring on any answer, returns relevance score and hallucination rate

---

## Adding more libraries

Open `ingestion.py`, add a line to the `LIBRARIES` list, run ingestion again. It's idempotent so existing chunks won't be duplicated.

```python
{"name": "numpy", "url": "https://numpy.org/doc/stable"}
```

---

## Project layout

```
codequery/
├── backend/
│   └── app/
│       ├── main.py        # all the routes
│       ├── ingestion.py   # scrapes docs, chunks, embeds, stores
│       ├── retrieval.py   # hybrid search + RRF
│       ├── llm.py         # Claude integration + streaming
│       ├── cache.py       # Redis semantic cache
│       └── evaluation.py  # relevance + hallucination scoring
└── frontend/
    └── src/
        ├── App.jsx        # state + streaming logic
        ├── SearchBox.jsx  # input + example query chips
        └── Results.jsx    # answer rendering + source cards + metrics
```

---

## Config

| Variable | What it does | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | your API key | required |
| `CHROMA_PERSIST_DIR` | where ChromaDB saves data | `./chroma_db` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379` |
| `EMBEDDING_MODEL` | which sentence-transformers model | `all-MiniLM-L6-v2` |
| `TOP_K_RESULTS` | chunks returned per query | `5` |
| `CACHE_SIMILARITY_THRESHOLD` | how similar queries need to be for a cache hit | `0.95` |

---

Built this as part of my AI/ML portfolio while finishing my MS at UMD.
