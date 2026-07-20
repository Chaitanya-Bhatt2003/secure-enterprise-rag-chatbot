"""Optional cross-encoder reranking.

A cross-encoder scores (query, chunk) PAIRS jointly, which is far more
accurate than the bi-encoder similarity used for the first-stage fetch,
at the cost of an extra model. Off by default (config.USE_RERANKER); if
the model can't be loaded we return the input order unchanged, so this
is never a hard dependency.

Reranking only reorders chunks that already passed the RBAC filter in
retriever.py — it can't (re)introduce forbidden content.
"""

from functools import lru_cache
from typing import List, Optional

from config import RERANKER_MODEL
from retrieval.retriever import RetrievedChunk


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load the cross-encoder once; None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL)
    except Exception:
        return None


def rerank(
    query: str, chunks: List[RetrievedChunk], top_k: Optional[int] = None
) -> List[RetrievedChunk]:
    """Rerank chunks by cross-encoder relevance to the query."""
    if not chunks:
        return chunks

    model = _get_cross_encoder()
    if model is None:
        return chunks if top_k is None else chunks[:top_k]

    scores = model.predict([(query, c.text) for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked if top_k is None else ranked[:top_k]
