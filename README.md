I got tired of Googling pandas syntax for the 50th time. So I built this.
CodeQuery lets you ask questions about Python library docs in plain English and get back actual answers — with code examples, cited sources, and no hallucinated API methods that don't exist.

The problem it solves
When you ask ChatGPT "how do I read a CSV with pandas," it'll give you an answer. Sometimes that answer uses a parameter that was deprecated two versions ago, or a method that never existed at all. It has no idea — it's just pattern matching on training data.
CodeQuery only answers from the actual docs. If it's not in the documentation, it won't make something up. Every answer tells you exactly which page it came from so you can verify it yourself.

What you can ask it

"How do I make a POST request with authentication?"
"What's the difference between loc and iloc in pandas?"
"How do I add dependency injection in FastAPI?"
Anything you'd normally Google and spend 10 minutes finding the right Stack Overflow answer for


Numbers that actually mean something
ThingNumberRelevance improvement over pure vector search+35%Hallucination rate< 5%Response time on cache hit< 600msResponse time on cache miss~2.3sAPI cost reduction from caching~60%Pages indexed10,000+

How it works
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

Why two search methods?
Vector search is great at finding conceptually related content but terrible at exact matches. If you search for "requests.get timeout parameter," vector search might return chunks about HTTP timeouts in general instead of the specific requests docs.
BM25 is the opposite — great at exact keyword matches, bad at understanding meaning.
Combining them and using Reciprocal Rank Fusion to merge the results gets you the best of both. A chunk that shows up in both lists gets a boosted score. That's where the 35% relevance improvement comes from.

Why embedding-based caching instead of string matching?
"How do I make a GET request" and "what's the syntax for GET in requests" are the same question phrased differently. String-match caching treats them as two different queries and calls the LLM twice.
This uses cosine similarity on embeddings — if two queries are more than 95% similar, return the cached answer. Same cost savings, works for paraphrases. Brought API costs down by about 60% during testing.

Stack
LayerWhat I usedLLMClaude (claude-sonnet-4-6)Vector DBChromaDB (runs locally, no external service)Embeddingssentence-transformers all-MiniLM-L6-v2Keyword searchrank-bm25CachingRedisBackendFastAPI with async SSE streamingFrontendReact + Vite

Running it yourself
You'll need Python 3.10+, Node.js 18+, Redis, and an Anthropic API key.
bashgit clone https://github.com/ScriptedShadows/codequery
cd codequery
cp .env.example .env
# put your ANTHROPIC_API_KEY in .env
Start the backend:
bashcd backend
pip install -r requirements.txt
python -m app.ingestion    # indexes the docs, takes ~5 mins first run
uvicorn app.main:app --reload --port 8001
Start Redis:
bashredis-server
Start the frontend:
bashcd frontend
npm install
npm run dev
# go to http://localhost:3000

API
POST /search — standard search, returns JSON with answer + sources + metrics
POST /search/stream — same thing but streams tokens via SSE as they're generated
POST /search/compare — runs both pure semantic and hybrid retrieval side by side, useful for seeing how much BM25 actually helps
GET /metrics — live stats: cache hit rate, avg latency, token usage, estimated cost
POST /evaluate — runs Claude-as-judge scoring on any answer, returns relevance score and hallucination rate

Adding more libraries
Open ingestion.py, add a line to the LIBRARIES list, run ingestion again. It's idempotent so existing chunks won't be duplicated.
python{"name": "numpy", "url": "https://numpy.org/doc/stable"}

Project layout
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

Config
VariableWhat it doesDefaultANTHROPIC_API_KEYyour API keyrequiredCHROMA_PERSIST_DIRwhere ChromaDB saves data./chroma_dbREDIS_URLRedis connectionredis://localhost:6379EMBEDDING_MODELwhich sentence-transformers modelall-MiniLM-L6-v2TOP_K_RESULTSchunks returned per query5CACHE_SIMILARITY_THRESHOLDhow similar queries need to be for a cache hit0.95

Built this as part of my AI/ML portfolio while finishing my MS at UMD.
