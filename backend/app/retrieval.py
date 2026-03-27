"""
Retrieval module for CodeQuery.

Hybrid search combining ChromaDB semantic search with BM25 keyword search,
fused using Reciprocal Rank Fusion (RRF).
"""

import logging
import os
import re
import threading
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K: int = int(os.getenv("TOP_K_RESULTS", "5"))
COLLECTION_NAME: str = "codequery_docs"

# How many candidates each retriever fetches before RRF merging
_RETRIEVER_TOP_K: int = 10

# Module-level singletons (initialized lazily)
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    """Return (and cache) the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'…", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    """Return (and cache) the ChromaDB collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------
class BM25Index:
    """Keyword search index built from ChromaDB-stored chunks.

    Loads all documents from the ChromaDB collection on first use, tokenizes
    them, and builds a BM25Okapi index. Thread-safe via a lock.
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._ids: list[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split on non-alphanumeric characters.

        Args:
            text: Raw text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build(self) -> None:
        """Load all chunks from ChromaDB and build the BM25 index."""
        collection = _get_collection()
        count = collection.count()
        if count == 0:
            logger.warning("ChromaDB collection is empty — BM25 index will be empty")
            self._index = BM25Okapi([[""]])
            return

        # Fetch all documents in one call
        result = collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        self._ids = result["ids"]
        self._documents = result["documents"]
        self._metadatas = result["metadatas"]

        tokenized_corpus = [self._tokenize(doc) for doc in self._documents]
        self._index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(self._documents))

    def search(self, query: str, top_k: int = _RETRIEVER_TOP_K) -> list[dict[str, Any]]:
        """Search the BM25 index for the most relevant chunks.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return.

        Returns:
            Ranked list of dicts with ``id``, ``text``, ``metadata``, and ``score``.
        """
        with self._lock:
            if self._index is None:
                self._build()

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top_k indices sorted by descending score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            results.append(
                {
                    "id": self._ids[idx],
                    "text": self._documents[idx],
                    "metadata": self._metadatas[idx],
                    "score": float(scores[idx]),
                }
            )

        logger.debug(
            "BM25 search for '%s': top scores = %s",
            query[:60],
            [round(r["score"], 4) for r in results[:5]],
        )
        return results

    def rebuild(self) -> None:
        """Force a rebuild of the BM25 index (e.g. after new ingestion)."""
        with self._lock:
            self._index = None
            self._documents = []
            self._metadatas = []
            self._ids = []
        logger.info("BM25 index invalidated — will rebuild on next query")


# Module-level BM25 singleton
_bm25_index = BM25Index()


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------
def semantic_search(query: str, top_k: int = _RETRIEVER_TOP_K) -> list[dict[str, Any]]:
    """Run a vector-similarity search against ChromaDB.

    Args:
        query: The natural-language search query.
        top_k: Number of results to return.

    Returns:
        A list of dicts with ``id``, ``text``, ``metadata``, and ``score``.
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0] if results["ids"] else []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    hits: list[dict[str, Any]] = []
    for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        hits.append(
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # cosine distance → similarity
            }
        )

    logger.info("Semantic search for '%s' returned %d hits", query[:60], len(hits))
    return hits


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    semantic_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Merge two ranked result lists using Reciprocal Rank Fusion.

    For each result, RRF score = 1 / (rank + k), summed across both lists.
    Results appearing in both lists get the sum of their individual RRF scores.

    Args:
        semantic_results: Ranked results from semantic search.
        bm25_results: Ranked results from BM25 keyword search.
        k: RRF constant (default 60, per the original RRF paper).

    Returns:
        Merged list sorted by combined RRF score (descending).
    """
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(semantic_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + k)
        chunk_data[doc_id] = result

    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + k)
        if doc_id not in chunk_data:
            chunk_data[doc_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    merged: list[dict[str, Any]] = []
    for doc_id in sorted_ids:
        entry = chunk_data[doc_id].copy()
        entry["score"] = rrf_scores[doc_id]
        merged.append(entry)

    logger.debug(
        "RRF fusion: %d semantic + %d bm25 → %d merged (top score %.6f)",
        len(semantic_results),
        len(bm25_results),
        len(merged),
        merged[0]["score"] if merged else 0.0,
    )
    return merged


# ---------------------------------------------------------------------------
# Top-level retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    top_k: int = TOP_K,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Retrieve the most relevant documentation chunks for a query.

    Args:
        query: The user's natural-language query.
        top_k: Number of final results to return.
        mode: Retrieval mode — ``"hybrid"`` (default), or ``"semantic"`` only.

    Returns:
        Ranked list of chunks, each with ``text``, ``metadata``, ``score``,
        and a top-level ``retrieval_mode`` key on each result.
    """
    if mode == "semantic":
        results = semantic_search(query, top_k=top_k)
        for r in results:
            r["retrieval_mode"] = "semantic"
        return results

    # Hybrid: semantic + BM25 → RRF
    sem_results = semantic_search(query, top_k=_RETRIEVER_TOP_K)
    bm25_results = _bm25_index.search(query, top_k=_RETRIEVER_TOP_K)

    merged = reciprocal_rank_fusion(sem_results, bm25_results)
    final = merged[:top_k]

    for r in final:
        r["retrieval_mode"] = "hybrid"

    logger.info(
        "Hybrid retrieval for '%s': %d semantic + %d bm25 → %d final (RRF)",
        query[:60],
        len(sem_results),
        len(bm25_results),
        len(final),
    )
    return final
