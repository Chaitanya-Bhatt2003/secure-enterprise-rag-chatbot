"""PII / confidential-data masking.

Detection engines, in order of preference:
  1. Microsoft Presidio (if installed AND its spaCy model is available)
  2. A regex fallback covering SSNs, credit cards, emails, phone numbers,
     and salary/compensation figures.

The fallback means masking is ALWAYS active — a missing optional
dependency degrades accuracy, never disables the control.

WHERE masking runs matters as much as how: chain.py calls
`mask_chunks()` on retrieved text BEFORE it is inserted into the LLM
prompt. The raw PII therefore never reaches the model for non-admin
users, so it cannot leak through paraphrase, translation, or clever
prompting — the model simply never saw it. Output-side filtering in
guardrails.py is a second net, not the primary control.

Role policy (ROLE_VISIBLE_ENTITIES): admins see everything unmasked;
HR may see salary figures (their job requires it) but not credit cards;
everyone else gets all recognized PII types masked.
"""

import re
from typing import List, Tuple

from retrieval.retriever import RetrievedChunk

# Entity types this module recognizes.
SSN = "US_SSN"
CREDIT_CARD = "CREDIT_CARD"
EMAIL = "EMAIL_ADDRESS"
PHONE = "PHONE_NUMBER"
SALARY = "SALARY_FIGURE"

# Which roles may see which entity types UNMASKED. Anything not listed
# for a role is masked — an allowlist, so new entity types are masked by
# default for everyone but admin (fail closed).
ROLE_VISIBLE_ENTITIES = {
    "admin": {SSN, CREDIT_CARD, EMAIL, PHONE, SALARY},
    "hr": {SALARY, EMAIL, PHONE},   # HR legitimately works with comp data
    "finance": {SALARY},            # finance sees budget/comp figures
    "engineering": set(),
    "general": set(),
}

# ---------------------------------------------------------------------------
# Regex fallback patterns
# ---------------------------------------------------------------------------
_REGEX_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # 123-45-6789 (labeled-context variants also match the bare pattern)
    (SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # 13-19 digit card numbers, with optional space/dash separators
    (CREDIT_CARD, re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    (EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # +1 (555) 123-4567 / 555-123-4567 / +91 98765 43210
    (PHONE, re.compile(r"(?<![\d-])(?:\+?\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?\d{3,5}[ -]\d{3,5}(?:[ -]\d{2,5})?(?![\d-])")),
    # $85,000 / $1.2M / ₹12,00,000 — currency amounts (comp/budget figures)
    (SALARY, re.compile(r"[$₹€£]\s?\d[\d,]*(?:\.\d+)?\s?(?:[MKmk]\b|million\b|lakh\b|crore\b)?")),
]


def _presidio_analyzer():
    """Build a Presidio analyzer once, or None if unavailable."""
    global _PRESIDIO
    try:
        return _PRESIDIO
    except NameError:
        pass
    try:
        from presidio_analyzer import AnalyzerEngine
        _PRESIDIO = AnalyzerEngine()
        # Smoke-test: fails here (not mid-request) if the spaCy model is missing.
        _PRESIDIO.analyze(text="test", language="en")
    except Exception:
        _PRESIDIO = None
    return _PRESIDIO


def _detect_presidio(text: str) -> List[Tuple[str, int, int]]:
    """(entity_type, start, end) spans found by Presidio."""
    analyzer = _presidio_analyzer()
    if analyzer is None:
        return []
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=["US_SSN", "CREDIT_CARD", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        score_threshold=0.4,
    )
    return [(r.entity_type, r.start, r.end) for r in results]


def _detect_regex(text: str) -> List[Tuple[str, int, int]]:
    """(entity_type, start, end) spans found by the regex fallback."""
    spans = []
    for entity_type, pattern in _REGEX_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((entity_type, match.start(), match.end()))
    return spans


def mask_text(text: str, role: str) -> Tuple[str, List[str]]:
    """Mask PII in `text` that the given role may not see.

    Returns (masked_text, sorted list of entity types that were masked).
    Unknown roles get the empty allowlist — fail closed.
    """
    visible = ROLE_VISIBLE_ENTITIES.get(role, set())

    # Presidio results take precedence; regex fills gaps (e.g. salary
    # figures, which Presidio has no built-in recognizer for).
    spans = _detect_presidio(text) + _detect_regex(text)
    spans = [(etype, s, e) for etype, s, e in spans if etype not in visible]
    if not spans:
        return text, []

    # Merge overlapping spans into one redaction. Detectors can flag
    # overlapping regions — e.g. the phone-number pattern matches the first
    # digits of a credit card. If we redacted the shorter span alone, the
    # rest of the card (its trailing digits) would survive. Merging the union
    # guarantees no PII tail is left behind; the label comes from the longest
    # contributing span (the full credit card, not the partial phone match).
    spans.sort(key=lambda x: (x[1], x[2]))
    merged: List[List] = []  # [start, end, label, label_len]
    for etype, start, end in spans:
        length = end - start
        if merged and start <= merged[-1][1]:
            group = merged[-1]
            group[1] = max(group[1], end)
            if length > group[3]:
                group[2], group[3] = etype, length
        else:
            merged.append([start, end, etype, length])

    # Replace right-to-left so earlier offsets stay valid (merged spans are
    # sorted ascending and non-overlapping).
    masked_types = set()
    result = text
    for start, end, etype, _ in reversed(merged):
        result = result[:start] + f"[REDACTED:{etype}]" + result[end:]
        masked_types.add(etype)
    return result, sorted(masked_types)


def mask_chunks(chunks: List[RetrievedChunk], role: str) -> Tuple[List[RetrievedChunk], List[str]]:
    """Mask every retrieved chunk before it goes anywhere near a prompt.

    Returns new RetrievedChunk objects (originals untouched) and the union
    of masked entity types, for the audit log and UI badge.
    """
    masked_chunks = []
    all_masked: set = set()
    for chunk in chunks:
        masked_text, masked_types = mask_text(chunk.text, role)
        masked_chunks.append(
            RetrievedChunk(text=masked_text, score=chunk.score, metadata=chunk.metadata)
        )
        all_masked.update(masked_types)
    return masked_chunks, sorted(all_masked)
