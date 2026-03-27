"""
Evaluation framework for CodeQuery.

Uses Claude-as-a-judge to score answer relevance and detect hallucinations.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
MODEL: str = "claude-sonnet-4-6"

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Return (and cache) the async Anthropic client."""
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
        _client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _format_context(context_chunks: list[dict[str, Any] | str]) -> str:
    """Format context chunks into a readable block.

    Args:
        context_chunks: List of chunk dicts (with ``text`` key) or raw strings.

    Returns:
        Numbered context string.
    """
    lines: list[str] = []
    for i, chunk in enumerate(context_chunks):
        text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
        lines.append(f"[Context {i}]: {text}")
    return "\n\n".join(lines)


async def evaluate_relevance(
    query: str,
    answer: str,
    context_chunks: list[dict[str, Any] | str],
) -> float:
    """Score the relevance of an answer to the query given the context.

    Uses Claude to grade relevance on a 0.0–1.0 scale.

    Args:
        query: The original user question.
        answer: The generated answer.
        context_chunks: The context passages used for generation.

    Returns:
        A relevance score between 0.0 and 1.0.
    """
    client = _get_client()
    context = _format_context(context_chunks)

    prompt = (
        "You are an evaluation judge. Given a query, retrieved context passages, "
        "and a generated answer, rate the answer's relevance from 0.0 to 1.0.\n\n"
        "Consider:\n"
        "- Does the answer address the query directly?\n"
        "- Does the answer use the provided context appropriately?\n"
        "- Is the answer accurate based on the context?\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        'Return ONLY a JSON object: {{"score": 0.85, "reason": "..."}}'
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        result = json.loads(raw)
        score = float(result["score"])
        score = max(0.0, min(1.0, score))
        logger.info(
            "Relevance evaluation: %.2f — %s", score, result.get("reason", "")
        )
        return score
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Failed to parse relevance response: %s — raw: %s", exc, raw)
        return 0.0


async def evaluate_hallucination(
    answer: str,
    context_chunks: list[dict[str, Any] | str],
) -> float:
    """Detect hallucinations — claims not supported by the context.

    Uses Claude to estimate the fraction of unsupported claims.

    Args:
        answer: The generated answer to evaluate.
        context_chunks: The context passages that were available.

    Returns:
        A hallucination rate between 0.0 (none) and 1.0 (fully hallucinated).
    """
    client = _get_client()
    context = _format_context(context_chunks)

    prompt = (
        "You are a hallucination detector. Given context passages and a generated "
        "answer, identify any claims in the answer that are NOT supported by the "
        "context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        "Return ONLY a JSON object: "
        '{{"hallucination_rate": 0.1, "unsupported_claims": ["claim1", ...]}}'
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        result = json.loads(raw)
        rate = float(result["hallucination_rate"])
        rate = max(0.0, min(1.0, rate))
        claims = result.get("unsupported_claims", [])
        logger.info(
            "Hallucination evaluation: %.2f — unsupported claims: %s", rate, claims
        )
        return rate
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error(
            "Failed to parse hallucination response: %s — raw: %s", exc, raw
        )
        return 0.0


async def run_evaluation(
    query: str,
    answer: str,
    sources: list[dict[str, Any] | str],
) -> dict[str, Any]:
    """Run the full evaluation suite on a query/answer pair.

    Args:
        query: The original user question.
        answer: The generated answer.
        sources: The context chunks or source strings.

    Returns:
        Dict with ``relevance_score``, ``hallucination_rate``, and ``evaluated_at``.
    """
    relevance = await evaluate_relevance(query, answer, sources)
    hallucination = await evaluate_hallucination(answer, sources)

    result = {
        "relevance_score": round(relevance, 4),
        "hallucination_rate": round(hallucination, 4),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Evaluation complete: %s", result)
    return result
