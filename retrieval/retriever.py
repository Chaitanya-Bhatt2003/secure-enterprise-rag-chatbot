"""RBAC-aware retrieval.

THE CORE SECURITY PROPERTY of this module: the role filter is passed as a
`where` clause INSIDE every ChromaDB query, so chunks the user's role is
not allowed to see are excluded by the database BEFORE similarity ranking.
There is no code path that fetches unrestricted results and filters them
afterwards — a restricted chunk can never appear in a candidate list,
a debug log, or an LLM prompt for an unauthorized user.

Supports plain vector search and optional hybrid search (BM25 keyword
scores fused with vector scores via Reciprocal Rank Fusion). The BM25 index
is built ONLY over documents already passing the same role filter, so
hybrid mode preserves the pre-filtering guarantee.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from config import CANDIDATE_K, ROLES, TOP_K, USE_HYBRID_SEARCH
from retrieval.vector_store import get_collection, get_embedding_model


@dataclass
class RetrievedChunk:
    """One retrieved chunk plus everything needed for citations/auditing."""
    text: str
    score: float                      # higher = more relevant
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source_file", "unknown")

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0))


def _role_filter(role: str) -> dict:
    """Build the Chroma `where` clause enforcing RBAC.

    Raises on unknown roles instead of defaulting to something permissive —
    fail CLOSED, not open.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role!r}")
    return {f"role_{role}": True}


def _vector_search(query: str, role: str, k: int) -> List[RetrievedChunk]:
    """Similarity search with the RBAC filter applied inside the query."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    embedding = get_embedding_model().encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, collection.count()),
        where=_role_filter(role),  # <-- pre-filter: DB never ranks forbidden chunks
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Cosine distance -> similarity in [0, 1]-ish for readability.
        chunks.append(RetrievedChunk(text=text, score=1.0 - dist, metadata=meta or {}))
    return chunks


def _bm25_search(query: str, role: str, k: int) -> List[RetrievedChunk]:
    """Keyword (BM25) search over ONLY the chunks the role may access.

    Note the corpus itself is fetched with the same `where` role filter, so
    the keyword index never even contains forbidden text for this query.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []  # hybrid gracefully degrades to vector-only

    collection = get_collection()
    allowed = collection.get(where=_role_filter(role), include=["documents", "metadatas"])
    docs = allowed.get("documents") or []
    if not docs:
        return []
    metas = allowed.get("metadatas") or [{}] * len(docs)

    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        RetrievedChunk(text=docs[i], score=float(scores[i]), metadata=metas[i] or {})
        for i in ranked
        if scores[i] > 0
    ]


def _reciprocal_rank_fusion(
    result_lists: List[List[RetrievedChunk]], k: int, rrf_k: int = 60
) -> List[RetrievedChunk]:
    """Fuse ranked lists by RRF; dedupe on chunk text."""
    fused: dict[str, RetrievedChunk] = {}
    fused_scores: dict[str, float] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            key = chunk.text
            fused.setdefault(key, chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    ordered = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
    out = []
    for key in ordered:
        chunk = fused[key]
        chunk.score = fused_scores[key]
        out.append(chunk)
    return out


def retrieve(
    query: str,
    role: str,
    top_k: int = TOP_K,
    hybrid: Optional[bool] = None,
) -> List[RetrievedChunk]:
    """Retrieve the top_k chunks the given role is allowed to see.

    hybrid=None uses the config default; True fuses BM25 + vector results.
    Every underlying search path applies the role filter at the database
    level (see module docstring).
    """
    use_hybrid = USE_HYBRID_SEARCH if hybrid is None else hybrid

    vector_results = _vector_search(query, role, CANDIDATE_K)
    if not use_hybrid:
        return vector_results[:top_k]

    bm25_results = _bm25_search(query, role, CANDIDATE_K)
    if not bm25_results:
        return vector_results[:top_k]

    return _reciprocal_rank_fusion([vector_results, bm25_results], top_k)
