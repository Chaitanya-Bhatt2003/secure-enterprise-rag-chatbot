"""Central configuration for the Secure Enterprise RAG Chatbot.

Everything tunable lives here: model names, chunking parameters, the role
model used for RBAC, and the mock user database. Secrets (the Groq API key)
are loaded from a local .env file via python-dotenv and are NEVER hardcoded.
"""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root so the app works no matter the CWD.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Secrets (from .env) ---------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Models ----------------------------------------------------------------
LLM_MODEL = "llama-3.3-70b-versatile"          # served by Groq
EMBEDDING_MODEL = "all-MiniLM-L6-v2"           # sentence-transformers, 384-dim
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional reranker

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval -------------------------------------------------------------
TOP_K = 4                  # chunks handed to the LLM
CANDIDATE_K = 12           # candidates fetched before rerank/hybrid fusion
USE_HYBRID_SEARCH = True   # BM25 + vector fusion (falls back to vector-only)
USE_RERANKER = False       # cross-encoder rerank (needs extra model download)

# --- Storage ---------------------------------------------------------------
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "enterprise_docs"
AUDIT_LOG_FILE = str(PROJECT_ROOT / "logs" / "audit.log")
SAMPLE_DOCS_DIR = str(PROJECT_ROOT / "sample_docs")

# --- RBAC role model -------------------------------------------------------
# Every chunk stores one boolean metadata flag per role (role_admin, role_hr,
# ...). Retrieval filters on the *current* user's flag INSIDE the Chroma
# query, so restricted chunks are excluded before similarity ranking ever
# happens — they can never appear in a result set, let alone a prompt.
ROLES = ["admin", "hr", "finance", "engineering", "general"]

SENSITIVITY_LEVELS = ["public", "internal", "confidential", "restricted"]

DEPARTMENTS = ["hr", "finance", "engineering", "general"]


def _sha256(password: str) -> str:
    """Hash a password. Mock-auth only — a real deployment would use bcrypt
    with per-user salts and an identity provider (SSO/OIDC)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# Mock user database: username -> {password_hash, role}.
# Plaintext passwords (for the demo): see README.
MOCK_USERS = {
    "admin":  {"password_hash": _sha256("admin123"), "role": "admin"},
    "hannah": {"password_hash": _sha256("hr123"),    "role": "hr"},
    "frank":  {"password_hash": _sha256("fin123"),   "role": "finance"},
    "erin":   {"password_hash": _sha256("eng123"),   "role": "engineering"},
    "guest":  {"password_hash": _sha256("guest123"), "role": "general"},
}
