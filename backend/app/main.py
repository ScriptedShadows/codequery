"""
FastAPI application for CodeQuery.

Exposes search endpoints with semantic caching, streaming SSE responses,
evaluation scoring, and live metrics.
"""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.retrieval import retrieve
from app.llm import generate_answer, generate_answer_stream
from app.cache import semantic_cache
from app.evaluation import run_evaluation

# Lazy import — embedding model loaded once via retrieval module
from app.retrieval import _get_model as _get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeQuery",
    description="AI-powered Python documentation search with citations",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    """Payload for the /search endpoint."""

    query: str = Field(..., min_length=1, max_length=500, description="Natural-language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    stream: bool = Field(default=False, description="Whether to stream the response")


class SourceInfo(BaseModel):
    """A single retrieved source chunk returned in the response."""

    text: str
    library: str
    page_title: str
    source_url: str
    score: float


class SearchResponse(BaseModel):
    """Structured JSON response for /search."""

    answer: str
    sources: list[SourceInfo]
    metrics: dict[str, Any]


class EvaluateRequest(BaseModel):
    """Payload for the /evaluate endpoint."""

    query: str = Field(..., min_length=1, description="The original question")
    answer: str = Field(..., min_length=1, description="The generated answer")
    sources: list[str] = Field(..., description="Context source texts used for generation")


class EvaluateResponse(BaseModel):
    """Response from the /evaluate endpoint."""

    relevance_score: float
    hallucination_rate: float
    evaluated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_sources(chunks: list[dict[str, Any]]) -> list[SourceInfo]:
    """Convert retrieval chunks to SourceInfo list.

    Args:
        chunks: Raw retrieval results.

    Returns:
        List of SourceInfo models with truncated text.
    """
    return [
        SourceInfo(
            text=c["text"][:300],
            library=c["metadata"].get("library", ""),
            page_title=c["metadata"].get("page_title", ""),
            source_url=c["metadata"].get("source_url", ""),
            score=round(c["score"], 4),
        )
        for c in chunks
    ]


def _embed_query(query: str) -> list[float]:
    """Embed a query string using the shared sentence-transformer model.

    Args:
        query: Text to embed.

    Returns:
        Embedding as a list of floats.
    """
    model = _get_embedding_model()
    return model.encode(query).tolist()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse | StreamingResponse:
    """Search Python library docs and get a Claude-generated answer.

    Checks the semantic cache first. On a miss, runs hybrid retrieval +
    Claude generation and stores the result in cache.
    """
    start = time.perf_counter()
    semantic_cache.record_query()

    # --- Cache check ---
    query_embedding = await asyncio.to_thread(_embed_query, request.query)
    cached = await asyncio.to_thread(semantic_cache.get, query_embedding)

    if cached is not None:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        semantic_cache.record_latency(elapsed_ms)
        logger.info("Returning cached response in %dms", elapsed_ms)
        cached_metrics = cached.get("metrics", {})
        cached_metrics["cache_hit"] = True
        cached_metrics["latency_ms"] = elapsed_ms
        return SearchResponse(
            answer=cached["answer"],
            sources=[SourceInfo(**s) for s in cached["sources"]],
            metrics=cached_metrics,
        )

    # --- Retrieval (sync ChromaDB → thread pool) ---
    try:
        chunks = await asyncio.to_thread(retrieve, request.query, request.top_k)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail="Retrieval failed") from exc

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    sources = _build_sources(chunks)

    # --- Streaming path (delegates to /search/stream logic internally) ---
    if request.stream:
        source_dicts = [s.model_dump() for s in sources]
        extra_metrics = {"cache_hit": False, "retrieval_mode": "hybrid"}

        async def _event_stream():
            full_answer_parts: list[str] = []
            async for sse_line in generate_answer_stream(
                request.query, chunks, source_dicts, extra_metrics
            ):
                # Collect answer tokens for caching
                try:
                    payload = json.loads(sse_line.removeprefix("data: ").strip())
                    if "token" in payload:
                        full_answer_parts.append(payload["token"])
                    if payload.get("done"):
                        # Cache the assembled response
                        usage = payload.get("usage", {})
                        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                        semantic_cache.record_tokens(total_tokens)
                        elapsed = round((time.perf_counter() - start) * 1000)
                        semantic_cache.record_latency(elapsed)
                        cache_resp = {
                            "answer": "".join(full_answer_parts),
                            "sources": source_dicts,
                            "metrics": {"retrieval_mode": "hybrid", **usage},
                        }
                        await asyncio.to_thread(
                            semantic_cache.set, query_embedding, cache_resp
                        )
                except (json.JSONDecodeError, AttributeError):
                    pass
                yield sse_line

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # --- Non-streaming path ---
    try:
        result = await generate_answer(request.query, chunks)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Answer generation failed") from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000)

    total_tokens = result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
    semantic_cache.record_tokens(total_tokens)
    semantic_cache.record_latency(elapsed_ms)

    # Store in cache
    source_dicts = [s.model_dump() for s in sources]
    cache_resp = {
        "answer": result["answer"],
        "sources": source_dicts,
        "metrics": {
            "retrieval_mode": "hybrid",
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
        },
    }
    await asyncio.to_thread(semantic_cache.set, query_embedding, cache_resp)

    return SearchResponse(
        answer=result["answer"],
        sources=sources,
        metrics={
            "latency_ms": elapsed_ms,
            "cache_hit": False,
            "retrieval_mode": "hybrid",
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
        },
    )


# ---------------------------------------------------------------------------
# Dedicated streaming endpoint
# ---------------------------------------------------------------------------
@app.post("/search/stream")
async def search_stream(request: SearchRequest) -> StreamingResponse:
    """Stream search results as SSE events.

    Cache hits return a single instant event. Cache misses stream Claude's
    response token-by-token, then cache the assembled answer.
    """
    start = time.perf_counter()
    semantic_cache.record_query()

    # --- Cache check ---
    query_embedding = await asyncio.to_thread(_embed_query, request.query)
    cached = await asyncio.to_thread(semantic_cache.get, query_embedding)

    if cached is not None:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        semantic_cache.record_latency(elapsed_ms)
        cached["metrics"] = {**cached.get("metrics", {}), "cache_hit": True, "latency_ms": elapsed_ms}
        cached["done"] = True

        async def _cached_stream():
            yield f"data: {json.dumps(cached)}\n\n"

        return StreamingResponse(_cached_stream(), media_type="text/event-stream")

    # --- Retrieval ---
    try:
        chunks = await asyncio.to_thread(retrieve, request.query, request.top_k)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail="Retrieval failed") from exc

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    sources = _build_sources(chunks)
    source_dicts = [s.model_dump() for s in sources]
    extra_metrics = {"cache_hit": False, "retrieval_mode": "hybrid"}

    async def _event_stream():
        full_answer_parts: list[str] = []
        async for sse_line in generate_answer_stream(
            request.query, chunks, source_dicts, extra_metrics
        ):
            try:
                payload = json.loads(sse_line.removeprefix("data: ").strip())
                if "token" in payload:
                    full_answer_parts.append(payload["token"])
                if payload.get("done"):
                    usage = payload.get("usage", {})
                    total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    semantic_cache.record_tokens(total_tokens)
                    elapsed = round((time.perf_counter() - start) * 1000)
                    semantic_cache.record_latency(elapsed)
                    cache_resp = {
                        "answer": "".join(full_answer_parts),
                        "sources": source_dicts,
                        "metrics": {"retrieval_mode": "hybrid", **usage},
                    }
                    await asyncio.to_thread(
                        semantic_cache.set, query_embedding, cache_resp
                    )
            except (json.JSONDecodeError, AttributeError):
                pass
            yield sse_line

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Comparison endpoint — semantic vs hybrid side by side
# ---------------------------------------------------------------------------
class CompareRequest(BaseModel):
    """Payload for the /search/compare endpoint."""

    query: str = Field(..., min_length=1, max_length=500, description="Natural-language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve per mode")


class CompareResultSide(BaseModel):
    """One side (semantic or hybrid) of the comparison."""

    answer: str
    sources: list[SourceInfo]
    latency_ms: int


class CompareResponse(BaseModel):
    """Side-by-side comparison of semantic vs hybrid retrieval."""

    query: str
    semantic: CompareResultSide
    hybrid: CompareResultSide


@app.post("/search/compare", response_model=CompareResponse)
async def search_compare(request: CompareRequest) -> CompareResponse:
    """Compare pure semantic search vs hybrid (semantic + BM25 + RRF).

    Runs both retrieval modes, generates Claude answers for each, and returns
    results side by side for relevance comparison.
    """
    # --- Semantic ---
    sem_start = time.perf_counter()
    sem_chunks = await asyncio.to_thread(retrieve, request.query, request.top_k, "semantic")
    if not sem_chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found (semantic)")
    sem_result = await generate_answer(request.query, sem_chunks)
    sem_ms = round((time.perf_counter() - sem_start) * 1000)

    # --- Hybrid ---
    hyb_start = time.perf_counter()
    hyb_chunks = await asyncio.to_thread(retrieve, request.query, request.top_k, "hybrid")
    if not hyb_chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found (hybrid)")
    hyb_result = await generate_answer(request.query, hyb_chunks)
    hyb_ms = round((time.perf_counter() - hyb_start) * 1000)

    return CompareResponse(
        query=request.query,
        semantic=CompareResultSide(
            answer=sem_result["answer"],
            sources=_build_sources(sem_chunks),
            latency_ms=sem_ms,
        ),
        hybrid=CompareResultSide(
            answer=hyb_result["answer"],
            sources=_build_sources(hyb_chunks),
            latency_ms=hyb_ms,
        ),
    )


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------
@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Return live cache, performance, and usage statistics."""
    stats = await asyncio.to_thread(semantic_cache.get_stats)
    return stats


# ---------------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------------
@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Run relevance and hallucination evaluation on a query/answer pair.

    Args:
        request: Contains the query, answer, and source texts.

    Returns:
        Scores for relevance and hallucination rate.
    """
    result = await run_evaluation(request.query, request.answer, request.sources)
    return EvaluateResponse(**result)
