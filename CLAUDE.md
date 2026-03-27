# CodeQuery — Project Context for Claude Code

## What This Project Is
A production-ready RAG system that lets developers search Python library documentation
using natural language queries. Think "AI-powered docs search" with citations, streaming
responses, and semantic caching.

## Tech Stack
- **Backend:** Python, FastAPI
- **LLM:** Anthropic Claude API — always use `claude-sonnet-4-6`
- **Vector DB:** ChromaDB (local, no external service)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Caching:** Redis (semantic caching via cosine similarity)
- **Frontend:** React with streaming responses
- **BM25:** rank-bm25 library
- **Search:** Hybrid retrieval = ChromaDB semantic search + BM25, fused with RRF

## Project Structure
```
codequery/
├── CLAUDE.md
├── .env.example
├── .env                  # never commit this
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py       # FastAPI app, routes
│       ├── ingestion.py  # scrape docs, chunk, embed, store in ChromaDB
│       ├── retrieval.py  # hybrid search: semantic + BM25 + RRF
│       ├── llm.py        # Claude API integration, streaming, prompt templates
│       └── cache.py      # Redis semantic caching logic
└── frontend/
    └── src/
        ├── App.jsx
        ├── SearchBox.jsx
        └── Results.jsx   # streaming responses + citations
```

## Build Order — Follow This Strictly
1. **Phase 1** — ingestion.py + ChromaDB setup + /search endpoint + Claude integration
2. **Phase 2** — Add BM25 to retrieval.py, implement Reciprocal Rank Fusion
3. **Phase 3** — Redis semantic caching in cache.py, streaming, evaluation metrics
4. **Phase 4** — React frontend with streaming + citations + metrics dashboard

Do NOT jump ahead. Finish and test each phase before moving to the next.

## Coding Conventions
- All Python code must have **type hints** and **docstrings**
- Use **async/await** throughout FastAPI (no sync routes)
- All API keys and config go in **.env** — never hardcoded
- Use **logging** (not print statements) everywhere
- Handle errors explicitly — no bare `except:` clauses
- Return structured JSON responses with consistent shape:
  ```json
  { "answer": "...", "sources": [...], "metrics": { "latency_ms": 0, "cache_hit": false } }
  ```

## Environment Variables (from .env)
```
ANTHROPIC_API_KEY=
CHROMA_PERSIST_DIR=./chroma_db
REDIS_URL=redis://localhost:6379
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RESULTS=5
CACHE_SIMILARITY_THRESHOLD=0.95
```

## Key Implementation Details

### Chunking Strategy
- Chunk size: 500–800 tokens
- Overlap: 100 tokens between chunks
- Metadata per chunk: `{ source_url, library, page_title, chunk_index }`

### Hybrid Retrieval (Phase 2+)
- Run ChromaDB semantic search → get top 10
- Run BM25 keyword search → get top 10
- Merge using Reciprocal Rank Fusion (RRF): `score = 1 / (rank + 60)`
- Return top 5 final chunks

### Claude Prompt Rules
- Always instruct Claude to ONLY use the provided context
- Always instruct Claude to cite sources by chunk index
- Stream responses — never wait for full completion
- Model: `claude-sonnet-4-6`, max_tokens: 1024

### Semantic Caching (Phase 3+)
- Embed the user query with sentence-transformers
- Check Redis for stored embeddings with cosine similarity > 0.95
- Cache hit → return stored response immediately
- Cache miss → call Claude, store result + embedding in Redis

### Evaluation Metrics to Track
- Answer relevance (0–1, graded by Claude-as-judge)
- Hallucination rate (does answer stay within retrieved context?)
- Average response latency (ms)
- Cache hit rate (%)
- Cost per query (tokens used × price)

## Libraries to Index (Phase 1 — start with these 3)
- `requests` → https://requests.readthedocs.io
- `pandas` → https://pandas.pydata.org/docs
- `FastAPI` → https://fastapi.tiangolo.com

## What "Done" Looks Like for Each Phase
- **Phase 1 done:** Can run `python ingestion.py`, docs get stored in ChromaDB,
  POST /search returns an answer with citations in < 5s
- **Phase 2 done:** Hybrid search measurably improves relevance vs pure vector search
- **Phase 3 done:** Cache hit returns response in < 100ms, metrics endpoint works
- **Phase 4 done:** React UI streams responses, shows sources, displays latency

## Common Mistakes to Avoid
- Don't use sync ChromaDB client inside async FastAPI routes — use a thread pool
- Don't embed the full document — chunk first, then embed each chunk
- Don't store raw HTML in ChromaDB — strip tags, clean text first
- Don't forget CORS middleware in FastAPI for the React frontend
- Redis stores embeddings as JSON-serialized lists — deserialize before cosine math
