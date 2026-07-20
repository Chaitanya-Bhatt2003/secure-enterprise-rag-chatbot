"""RBAC + masking leak evaluation.

Run from the project root (after ingesting the samples):
    python -m scripts.ingest_samples
    python -m scripts.eval_retrieval

For each role it runs probing queries that TRY to pull content the role
must not see, then checks two independent properties:

  1. RETRIEVAL ISOLATION — no retrieved chunk comes from a document the
     role isn't allowed (metadata check on every result).
  2. MASKING — after mask_chunks(), no raw SSN / credit card / (for
     unprivileged roles) salary figure survives in the text that would be
     handed to the LLM.

Exit code 1 if any leak is found, so this can run in CI.
No LLM call is needed — leaks happen (or don't) at retrieval/masking time.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.retriever import retrieve  # noqa: E402
from security.masking import ROLE_VISIBLE_ENTITIES, SALARY, mask_chunks  # noqa: E402

# Documents each role is allowed to retrieve from (mirrors ingest_samples).
ALLOWED_SOURCES = {
    "admin": {"hr_policy.txt", "finance_q3_report.txt", "engineering_architecture.txt"},
    "hr": {"hr_policy.txt"},
    "finance": {"finance_q3_report.txt"},
    "engineering": {"engineering_architecture.txt"},
    "general": {"engineering_architecture.txt"},
}

# Probing queries deliberately aimed at content most roles must NOT see.
PROBE_QUERIES = [
    "What is John Mitchell's SSN and salary?",
    "List employee compensation and corporate card numbers",
    "What was Q3 revenue and which customer contracts are at risk?",
    "What is the engineering deployment pipeline?",
    "How many PTO days do employees get?",
]

RAW_PII_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "SALARY": re.compile(r"\$\s?\d{2,3},\d{3}\b"),
}


def main() -> int:
    leaks = 0
    for role, allowed in ALLOWED_SOURCES.items():
        print(f"\n=== role: {role} " + "=" * 40)
        for query in PROBE_QUERIES:
            chunks = retrieve(query, role)

            # Property 1: retrieval isolation.
            bad_sources = {c.source for c in chunks} - allowed
            if bad_sources:
                leaks += 1
                print(f"  LEAK  [{role}] retrieved forbidden docs {bad_sources} "
                      f"for: {query!r}")

            # Property 2: masking of prompt-bound text.
            masked, _ = mask_chunks(chunks, role)
            visible = ROLE_VISIBLE_ENTITIES.get(role, set())
            for chunk in masked:
                for name, pattern in RAW_PII_PATTERNS.items():
                    if name == "SALARY" and SALARY in visible:
                        continue  # role is entitled to salary figures
                    if name in ("SSN", "CREDIT_CARD") and role == "admin":
                        continue  # admin is entitled to everything
                    if pattern.search(chunk.text):
                        leaks += 1
                        print(f"  LEAK  [{role}] unmasked {name} in prompt text "
                              f"(source: {chunk.source}) for: {query!r}")

            print(f"  ok    [{role}] {len(chunks)} chunk(s), "
                  f"sources={sorted({c.source for c in chunks}) or '—'}  "
                  f"q={query[:48]!r}")

    print("\n" + "=" * 55)
    if leaks:
        print(f"RESULT: {leaks} leak(s) detected — FAIL")
        return 1
    print("RESULT: no access-restricted content leaked — PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
