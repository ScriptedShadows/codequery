"""
Claude API integration for CodeQuery.

Builds a grounded prompt from retrieved chunks and streams Claude's response.
"""

import json
import logging
import os
from typing import Any, AsyncIterator

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
MODEL: str = "claude-sonnet-4-6"
MAX_TOKENS: int = 1024

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Return (and cache) the async Anthropic client."""
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
        _client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Args:
        chunks: List of retrieval results, each with ``text`` and ``metadata``.

    Returns:
        A formatted string of numbered source passages.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        lines.append(
            f"[Source {i}] (library: {meta.get('library', '?')}, "
            f"page: {meta.get('page_title', '?')}, "
            f"url: {meta.get('source_url', '?')})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(lines)


SYSTEM_PROMPT = (
    "You are CodeQuery, a helpful assistant that answers questions about Python "
    "library documentation. You MUST answer ONLY using the provided context passages. "
    "If the context does not contain enough information to answer, say so clearly. "
    "When you reference information, cite the source by its index (e.g. [Source 0]). "
    "Be concise and provide code examples when helpful."
)


def build_user_message(query: str, chunks: list[dict[str, Any]]) -> str:
    """Construct the user message with context and query.

    Args:
        query: The user's original question.
        chunks: Retrieved context chunks.

    Returns:
        The full user message string.
    """
    context = _build_context_block(chunks)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        "Answer the question using ONLY the context above. Cite sources by index."
    )


async def generate_answer(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a non-streaming answer from Claude.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.

    Returns:
        A dict with ``answer`` (str) and ``usage`` (dict with token counts).
    """
    client = _get_client()
    user_message = build_user_message(query, chunks)

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    logger.info(
        "Claude response: %d input tokens, %d output tokens",
        usage["input_tokens"],
        usage["output_tokens"],
    )
    return {"answer": answer, "usage": usage}


async def generate_answer_stream(
    query: str,
    chunks: list[dict[str, Any]],
    sources: list[dict[str, Any]] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Stream Claude's answer as SSE-compatible JSON events.

    Each yielded string is a complete SSE ``data:`` line. Token events carry
    ``{"token": "..."}``; the final event carries ``{"done": true, ...}``
    with sources, metrics, and the assembled full answer.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks (passed to the prompt).
        sources: Pre-built source dicts for the final event.
        extra_metrics: Additional metrics to include in the final event.

    Yields:
        SSE-formatted ``data: {...}\\n\\n`` strings.
    """
    client = _get_client()
    user_message = build_user_message(query, chunks)
    full_answer_parts: list[str] = []

    async with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text in stream.text_stream:
            full_answer_parts.append(text)
            yield f"data: {json.dumps({'token': text})}\n\n"

        # Gather final usage from the accumulated response
        response = await stream.get_final_message()

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    full_answer = "".join(full_answer_parts)

    final_payload: dict[str, Any] = {
        "done": True,
        "answer": full_answer,
        "usage": usage,
    }
    if sources is not None:
        final_payload["sources"] = sources
    if extra_metrics is not None:
        final_payload["metrics"] = {**extra_metrics, **usage}

    logger.info(
        "Streamed response: %d tokens in, %d tokens out",
        usage["input_tokens"],
        usage["output_tokens"],
    )
    yield f"data: {json.dumps(final_payload)}\n\n"
