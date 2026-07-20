"""Chunking: split loaded documents into overlapping chunks for embedding.

Uses RecursiveCharacterTextSplitter so splits prefer paragraph, then
sentence, then word boundaries. Overlap preserves context across chunk
borders (e.g. a salary table header and its rows).
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks, carrying over per-page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
