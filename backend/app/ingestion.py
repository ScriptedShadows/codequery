"""
Document ingestion pipeline for CodeQuery.

Scrapes Python library documentation, chunks the text, generates embeddings
via sentence-transformers, and stores everything in ChromaDB.
"""

import logging
import re
import hashlib
from typing import Optional

import chromadb
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE: int = 600          # target tokens per chunk (500-800 range)
CHUNK_OVERLAP: int = 100       # token overlap between consecutive chunks
COLLECTION_NAME: str = "codequery_docs"

# Documentation pages to scrape per library
LIBRARY_SOURCES: dict[str, list[str]] = {
    "requests": [
        "https://requests.readthedocs.io/en/latest/user/quickstart/",
        "https://requests.readthedocs.io/en/latest/user/advanced/",
        "https://requests.readthedocs.io/en/latest/api/",
    ],
    "pandas": [
        "https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html",
        "https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html",
        "https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html",
        "https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html",
        "https://pandas.pydata.org/docs/getting_started/intro_tutorials/05_add_columns.html",
        "https://pandas.pydata.org/docs/getting_started/intro_tutorials/06_calculate_statistics.html",
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html",
        "https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html",
    ],
    "fastapi": [
        "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "https://fastapi.tiangolo.com/tutorial/path-params/",
        "https://fastapi.tiangolo.com/tutorial/query-params/",
        "https://fastapi.tiangolo.com/tutorial/body/",
        "https://fastapi.tiangolo.com/tutorial/response-model/",
        "https://fastapi.tiangolo.com/tutorial/dependencies/",
    ],
}


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
def scrape_page(url: str) -> Optional[dict[str, str]]:
    """Fetch a documentation page and extract clean text content.

    Args:
        url: The documentation page URL to scrape.

    Returns:
        A dict with keys ``url``, ``title``, and ``text``, or ``None`` on failure.
    """
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "CodeQuery/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style/nav elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try to locate the main content area
    main = (
        soup.find("div", {"role": "main"})
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"body|content|main", re.I))
        or soup.body
    )

    if main is None:
        logger.warning("No content found for %s", url)
        return None

    text = main.get_text(separator="\n", strip=True)
    # Collapse excessive whitespace / blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    logger.info("Scraped %s — %d characters", url, len(text))
    return {"url": url, "title": title, "text": text}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _approx_token_count(text: str) -> int:
    """Rough token estimate: split on whitespace."""
    return len(text.split())


def chunk_text(
    text: str,
    source_url: str,
    library: str,
    page_title: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split *text* into overlapping chunks with metadata.

    Args:
        text: The full page text.
        source_url: Origin URL for the document.
        library: Library name (e.g. ``"pandas"``).
        page_title: Page title for metadata.
        chunk_size: Target token count per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        A list of dicts, each with ``text`` and ``metadata`` keys.
    """
    words = text.split()
    chunks: list[dict] = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            {
                "text": chunk_text_str,
                "metadata": {
                    "source_url": source_url,
                    "library": library,
                    "page_title": page_title,
                    "chunk_index": chunk_index,
                },
            }
        )
        chunk_index += 1
        start += chunk_size - overlap

    logger.info(
        "Chunked '%s' into %d chunks (avg ~%d tokens)",
        page_title,
        len(chunks),
        chunk_size,
    )
    return chunks


# ---------------------------------------------------------------------------
# Embedding + ChromaDB storage
# ---------------------------------------------------------------------------
def _chunk_id(metadata: dict, text: str) -> str:
    """Deterministic ID so re-ingestion is idempotent."""
    key = f"{metadata['source_url']}:{metadata['chunk_index']}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def ingest_library(
    library: str,
    urls: list[str],
    model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int:
    """Scrape, chunk, embed, and store docs for a single library.

    Args:
        library: Library name.
        urls: List of documentation URLs to process.
        model: The sentence-transformer model for embeddings.
        collection: The ChromaDB collection to upsert into.

    Returns:
        The number of chunks stored.
    """
    all_chunks: list[dict] = []

    for url in urls:
        page = scrape_page(url)
        if page is None:
            continue
        chunks = chunk_text(
            text=page["text"],
            source_url=page["url"],
            library=library,
            page_title=page["title"],
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks produced for library '%s'", library)
        return 0

    texts = [c["text"] for c in all_chunks]
    ids = [_chunk_id(c["metadata"], c["text"]) for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    logger.info("Generating embeddings for %d chunks (%s)…", len(texts), library)
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Stored %d chunks for '%s' in ChromaDB", len(texts), library)
    return len(texts)


def run_ingestion() -> None:
    """Run the full ingestion pipeline for all configured libraries."""
    logger.info("Loading embedding model '%s'…", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for library, urls in LIBRARY_SOURCES.items():
        count = ingest_library(library, urls, model, collection)
        total += count

    logger.info("Ingestion complete — %d total chunks stored.", total)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_ingestion()
