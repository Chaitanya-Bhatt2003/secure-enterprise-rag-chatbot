"""ChromaDB client + embedding model singletons.

Persistent local storage in ./chroma_db — no external services needed
besides the Groq API. Both the Chroma client and the sentence-transformers
model are cached module-level singletons because they are expensive to
construct and Streamlit re-runs the script on every interaction.
"""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    """Persistent Chroma client rooted at ./chroma_db."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


@lru_cache(maxsize=1)
def get_collection():
    """The single document collection, using cosine distance."""
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Shared sentence-transformers model (downloads on first use)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def reset_collection() -> None:
    """Delete every stored chunk so the next ingest starts from empty.

    Drops and recreates the collection, then clears the cached singleton so
    the next `get_collection()` call rebuilds it. Use this before a fresh
    ingest to avoid leftover documents from earlier runs/uploads polluting
    the database (which would otherwise show up as false leaks in the eval).
    """
    get_client().delete_collection(COLLECTION_NAME)
    get_collection.cache_clear()
    get_collection()  # recreate empty so callers can use it immediately


def collection_stats() -> dict:
    """Small helper for the UI: how many chunks are stored."""
    col = get_collection()
    return {"chunks": col.count()}
