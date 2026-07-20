"""The end-to-end secure RAG chain.

Pipeline order is the security design (each stage only ever sees what the
previous stage let through):

    1. guardrail INPUT check   — refuse prompt-injection attempts up front
    2. retrieve                — RBAC filter applied INSIDE the vector query
    3. (optional) rerank       — reorder allowed chunks only
    4. mask                    — PII redacted BEFORE prompt construction
    5. build prompt + call LLM — Groq / llama-3.3-70b-versatile
    6. guardrail OUTPUT check  — redact any PII that still slipped through
    7. audit log               — every query recorded, flagged or not

Because masking happens at step 4, the LLM never receives PII the user's
role may not see — so no paraphrasing trick at step 5 can leak it.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

import config
from llm.prompt_templates import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, format_context
from retrieval.retriever import RetrievedChunk, retrieve
from security.guardrails import REFUSAL_MESSAGE, check_input, check_output
from security.masking import mask_chunks
from utils.logger import audit_log


@dataclass
class ChainResult:
    """Everything the UI needs to render one answer."""
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)  # masked text
    masked_entities: List[str] = field(default_factory=list)
    injection_flagged: bool = False
    output_redacted: bool = False


@lru_cache(maxsize=1)
def _get_llm():
    """ChatGroq client singleton. Key comes from .env via config."""
    from langchain_groq import ChatGroq

    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatGroq(
        model=config.LLM_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,   # low temperature: factual, grounded answers
        max_tokens=1024,
    )


def answer_query(query: str, user: str, role: str, top_k: int = config.TOP_K) -> ChainResult:
    """Run one question through the full secure pipeline. See module docstring."""

    # 1. INPUT GUARDRAIL — refuse injection attempts before spending any
    # retrieval or LLM budget on them; log the attempt for the audit trail.
    input_check = check_input(query)
    if not input_check.is_safe:
        audit_log(user, role, query, sources=[], injection_flagged=True,
                  event="blocked_injection")
        return ChainResult(answer=REFUSAL_MESSAGE, injection_flagged=True)

    # 2. RETRIEVE — the role filter runs inside the Chroma query, so this
    # list can only ever contain chunks this role is allowed to see.
    chunks = retrieve(query, role, top_k=top_k)

    # 3. OPTIONAL RERANK (allowed chunks only).
    if config.USE_RERANKER and chunks:
        from retrieval.reranker import rerank
        chunks = rerank(query, chunks, top_k=top_k)

    if not chunks:
        audit_log(user, role, query, sources=[], event="no_context")
        return ChainResult(answer=NO_CONTEXT_ANSWER)

    # 4. MASK — PII is redacted from chunk text BEFORE the prompt is built.
    masked_chunks, masked_entities = mask_chunks(chunks, role)

    # 5. BUILD PROMPT + CALL LLM.
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=format_context(masked_chunks))),
        HumanMessage(content=query),
    ]
    response = _get_llm().invoke(messages)
    answer = response.content

    # 6. OUTPUT GUARDRAIL — belt-and-braces PII scan on the answer.
    output_check = check_output(answer, role)

    # 7. AUDIT — sources logged by name/metadata, never by content.
    source_records = [
        {
            "source_file": c.source,
            "page": c.page,
            "department": c.metadata.get("department"),
            "sensitivity_level": c.metadata.get("sensitivity_level"),
            "score": round(c.score, 4),
        }
        for c in masked_chunks
    ]
    audit_log(user, role, query, sources=source_records,
              masked_entities=masked_entities)

    return ChainResult(
        answer=output_check.text,
        sources=masked_chunks,
        masked_entities=masked_entities,
        output_redacted=bool(output_check.leaked_types),
    )
