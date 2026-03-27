"""
Redis semantic caching for CodeQuery.

Embeds incoming queries and checks Redis for previously cached responses
whose query embeddings have cosine similarity above a configurable threshold.
Gracefully degrades if Redis is unavailable.
"""

import json
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_SIMILARITY_THRESHOLD: float = float(
    os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95")
)
CACHE_TTL_SECONDS: int = 60 * 60 * 24  # 24 hours
LATENCY_LIST_MAX: int = 1000

# Redis key prefixes / names
_KEY_PREFIX = "codequery:cache:"
_STATS_HITS = "codequery:stats:hits"
_STATS_MISSES = "codequery:stats:misses"
_STATS_TOKENS = "codequery:stats:tokens"
_STATS_QUERIES = "codequery:stats:queries"
_LATENCIES_KEY = "codequery:latencies"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class SemanticCache:
    """Redis-backed semantic cache for query responses.

    Stores query embeddings alongside their full response dicts. On lookup,
    scans all cached embeddings and returns the response whose embedding is
    most similar to the query embedding — if it exceeds the threshold.

    If Redis is unreachable the cache silently no-ops so the application
    continues to function without caching.
    """

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self._client: Optional[redis.Redis] = None
        self._redis_url = redis_url
        self._connect()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Attempt to connect to Redis. Logs a warning on failure."""
        try:
            self._client = redis.from_url(
                self._redis_url, decode_responses=True, socket_connect_timeout=2
            )
            self._client.ping()
            logger.info("Connected to Redis at %s", self._redis_url)
        except (redis.ConnectionError, redis.TimeoutError, OSError) as exc:
            logger.warning("Redis unavailable (%s) — caching disabled", exc)
            self._client = None

    def _is_available(self) -> bool:
        """Check whether the Redis connection is alive."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError, OSError):
            logger.warning("Lost Redis connection — caching disabled")
            self._client = None
            return False

    # ------------------------------------------------------------------
    # Core cache operations
    # ------------------------------------------------------------------
    def get(
        self,
        query_embedding: list[float],
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
    ) -> Optional[dict[str, Any]]:
        """Look up a cached response by semantic similarity.

        Args:
            query_embedding: The embedding of the current query.
            threshold: Minimum cosine similarity to consider a hit.

        Returns:
            The cached response dict on a hit, or ``None`` on a miss.
        """
        if not self._is_available():
            return None

        try:
            keys = self._client.keys(f"{_KEY_PREFIX}*")
        except redis.RedisError as exc:
            logger.warning("Redis scan failed: %s", exc)
            return None

        best_score = -1.0
        best_response: Optional[dict[str, Any]] = None

        for key in keys:
            try:
                raw = self._client.get(key)
                if raw is None:
                    continue
                entry = json.loads(raw)
                stored_embedding = entry.get("embedding")
                if stored_embedding is None:
                    continue

                sim = _cosine_similarity(query_embedding, stored_embedding)
                if sim > best_score:
                    best_score = sim
                    best_response = entry.get("response")
            except (json.JSONDecodeError, redis.RedisError) as exc:
                logger.debug("Skipping cache key %s: %s", key, exc)
                continue

        if best_score >= threshold and best_response is not None:
            self._increment(_STATS_HITS)
            logger.info("Cache HIT (similarity=%.4f, threshold=%.2f)", best_score, threshold)
            return best_response

        self._increment(_STATS_MISSES)
        logger.info("Cache MISS (best similarity=%.4f, threshold=%.2f)", best_score, threshold)
        return None

    def set(
        self,
        query_embedding: list[float],
        response_dict: dict[str, Any],
    ) -> None:
        """Store a query embedding and its response in Redis.

        Args:
            query_embedding: The embedding of the query.
            response_dict: The full response to cache.
        """
        if not self._is_available():
            return

        key = f"{_KEY_PREFIX}{uuid4()}"
        entry = {
            "embedding": query_embedding,
            "response": response_dict,
            "cached_at": time.time(),
        }

        try:
            self._client.setex(key, CACHE_TTL_SECONDS, json.dumps(entry))
            logger.info("Cached response under %s (TTL=%ds)", key, CACHE_TTL_SECONDS)
        except redis.RedisError as exc:
            logger.warning("Failed to write cache: %s", exc)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _increment(self, key: str, amount: int = 1) -> None:
        """Safely increment a Redis counter."""
        if not self._is_available():
            return
        try:
            self._client.incrby(key, amount)
        except redis.RedisError:
            pass

    def record_latency(self, latency_ms: float) -> None:
        """Push a latency measurement onto the Redis list.

        Args:
            latency_ms: Request latency in milliseconds.
        """
        if not self._is_available():
            return
        try:
            self._client.lpush(_LATENCIES_KEY, str(latency_ms))
            self._client.ltrim(_LATENCIES_KEY, 0, LATENCY_LIST_MAX - 1)
        except redis.RedisError:
            pass

    def record_tokens(self, token_count: int) -> None:
        """Increment the global token usage counter.

        Args:
            token_count: Number of tokens to add.
        """
        self._increment(_STATS_TOKENS, token_count)

    def record_query(self) -> None:
        """Increment the global query counter."""
        self._increment(_STATS_QUERIES)

    def get_stats(self) -> dict[str, Any]:
        """Return cache and performance statistics.

        Returns:
            Dict with ``total_cached``, ``hit_count``, ``miss_count``,
            ``hit_rate``, latency percentiles, and token usage.
        """
        empty: dict[str, Any] = {
            "cache": {
                "total_cached": 0,
                "hit_count": 0,
                "miss_count": 0,
                "hit_rate": 0.0,
            },
            "performance": {
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            },
            "usage": {
                "total_queries": 0,
                "total_tokens_used": 0,
                "estimated_cost_usd": 0.0,
            },
        }

        if not self._is_available():
            return empty

        try:
            hits = int(self._client.get(_STATS_HITS) or 0)
            misses = int(self._client.get(_STATS_MISSES) or 0)
            total_cached = len(self._client.keys(f"{_KEY_PREFIX}*"))
            total_queries = int(self._client.get(_STATS_QUERIES) or 0)
            total_tokens = int(self._client.get(_STATS_TOKENS) or 0)

            total_lookups = hits + misses
            hit_rate = hits / total_lookups if total_lookups > 0 else 0.0

            # Latency stats
            raw_latencies = self._client.lrange(_LATENCIES_KEY, 0, -1)
            latencies = [float(v) for v in raw_latencies] if raw_latencies else []
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

            # Cost estimate (Claude Sonnet input ~$3/M, output ~$15/M — blended ~$6/M)
            estimated_cost = total_tokens * 6.0 / 1_000_000

            return {
                "cache": {
                    "total_cached": total_cached,
                    "hit_count": hits,
                    "miss_count": misses,
                    "hit_rate": round(hit_rate, 4),
                },
                "performance": {
                    "avg_latency_ms": round(avg_latency, 1),
                    "p95_latency_ms": round(p95_latency, 1),
                },
                "usage": {
                    "total_queries": total_queries,
                    "total_tokens_used": total_tokens,
                    "estimated_cost_usd": round(estimated_cost, 6),
                },
            }
        except redis.RedisError as exc:
            logger.warning("Failed to fetch stats: %s", exc)
            return empty


# Module-level singleton
semantic_cache = SemanticCache()
