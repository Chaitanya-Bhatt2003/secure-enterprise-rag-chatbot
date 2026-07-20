"""Guardrails: prompt-injection detection on input, leak detection on output.

INPUT SIDE — `check_input()` scans the user's question for known injection
phrasings ("ignore previous instructions", "you are now DAN", attempts to
extract the system prompt, etc.). Flagged input is refused outright rather
than "cleaned and forwarded": partially stripping an attack string still
leaves intent (and often working fragments) behind, so refusal is the
safer default. The flag is also written to the audit log.

OUTPUT SIDE — `check_output()` re-runs PII detection on the model's answer.
Because masking already ran on the context BEFORE the LLM saw it (see
masking.py), the model shouldn't know any secrets to leak — this is a
defense-in-depth net that catches (a) PII the input-side detector missed
in an odd format and (b) verbatim [REDACTED] markers being paraphrased
around. Any hit is redacted from the answer.
"""

import re
from dataclasses import dataclass
from typing import List

# Patterns that indicate an attempt to override instructions or exfiltrate
# the system prompt. Case-insensitive; kept deliberately readable so the
# list is easy to audit and extend.
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+|any\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|rules?)",
        r"forget\s+(all\s+|everything\s+)?(previous|prior|above|your)\s+(instructions?|rules?)?",
        r"(reveal|show|print|repeat|output|display)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",
        r"you\s+are\s+(now\s+)?(DAN|in\s+developer\s+mode|jailbroken)",
        r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?(unrestricted|unfiltered|uncensored)",
        r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(rules|restrictions|guidelines)",
        r"bypass\s+(your\s+)?(safety|security|content)\s+(filters?|guidelines?|restrictions?)",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*you\s+are",
        r"</?(system|instructions?)>",           # fake tag injection
        r"do\s+not\s+(mask|redact|censor|filter)",
        r"(without|remove|skip)\s+(the\s+)?(mask(ing)?|redact(ion)?s?)",
    ]
]

# PII shapes that must never appear in final output (mirrors masking.py's
# fallback patterns — kept independent so a masking bug doesn't take the
# output net down with it).
_OUTPUT_PII_PATTERNS = [
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
]


@dataclass
class InputCheck:
    is_safe: bool
    matched_patterns: List[str]


@dataclass
class OutputCheck:
    text: str            # possibly redacted answer
    leaked_types: List[str]


def check_input(user_query: str) -> InputCheck:
    """Screen a user question for prompt-injection attempts."""
    matched = [p.pattern for p in _INJECTION_PATTERNS if p.search(user_query)]
    return InputCheck(is_safe=not matched, matched_patterns=matched)


def check_output(answer: str, role: str) -> OutputCheck:
    """Screen the LLM answer for PII that slipped through.

    Admins are exempt (they're entitled to unmasked data); for everyone
    else, matching spans are replaced with [REDACTED:<type>].
    """
    if role == "admin":
        return OutputCheck(text=answer, leaked_types=[])

    leaked = []
    result = answer
    for entity_type, pattern in _OUTPUT_PII_PATTERNS:
        if pattern.search(result):
            leaked.append(entity_type)
            result = pattern.sub(f"[REDACTED:{entity_type}]", result)
    return OutputCheck(text=result, leaked_types=leaked)


REFUSAL_MESSAGE = (
    "⚠️ Your question was flagged by our security filters as a possible "
    "prompt-injection attempt and was not processed. Please rephrase your "
    "question. This event has been logged."
)
