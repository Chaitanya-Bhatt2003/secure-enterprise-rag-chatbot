"""Embedding + storage: turn tagged chunks into vectors in ChromaDB.

Uses sentence-transformers "all-MiniLM-L6-v2" (384-dim, fast on CPU) and
writes documents + embeddings + RBAC metadata into the persistent Chroma
collection in ./chroma_db.
"""

import hashlib
from typing import List

from langchain_core.documents import Document

from retrieval.vector_store import get_collection, get_embedding_model


def _chunk_id(chunk: Document, index: int) -> str:
    """Deterministic ID from source file + position + content hash, so
    re-ingesting the same file overwrites rather than duplicates chunks."""
    content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{chunk.metadata.get('source_file', 'unknown')}::{index}::{content_hash}"


def embed_and_store(chunks: List[Document]) -> int:
    """Embed chunks and upsert them into the Chroma collection.

    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    model = get_embedding_model()
    collection = get_collection()

    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)

    collection.upsert(
        ids=[_chunk_id(c, i) for i, c in enumerate(chunks)],
        embeddings=[e.tolist() for e in embeddings],
        documents=texts,
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def ingest_file(
    file_path: str,
    department: str,
    sensitivity_level: str,
    allowed_roles: List[str],
) -> int:
    """Full pipeline for one file: load -> chunk -> tag -> embed -> store."""
    from pathlib import Path

    from ingestion.chunker import chunk_documents
    from ingestion.loader import load_document
    from ingestion.metadata_tagger import tag_chunks

    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    tag_chunks(chunks, department, sensitivity_level, allowed_roles, Path(file_path).name)
    return embed_and_store(chunks)
