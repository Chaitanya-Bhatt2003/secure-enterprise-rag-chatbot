"""Document loading for the ingestion pipeline.

Maps file extensions to the appropriate LangChain document loader and
returns a list of `Document` objects (one per page for PDFs, one per row
for CSVs, one per file for TXT/DOCX).
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".md"}


def load_document(file_path: str) -> List[Document]:
    """Load a single file into LangChain Documents.

    Raises ValueError for unsupported extensions so callers can surface a
    clear error in the UI instead of a stack trace.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        # One Document per page; keeps `page` metadata for citations.
        loader = PyPDFLoader(str(path))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    elif ext in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    docs = loader.load()
    # Normalize the source metadata to just the file name (no local paths
    # leaking into citations shown to end users).
    for doc in docs:
        doc.metadata["source"] = path.name
    return docs
