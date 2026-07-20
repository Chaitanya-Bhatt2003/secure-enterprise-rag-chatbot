"""Ingest the three sample documents with their RBAC metadata.

Run from the project root:  python -m scripts.ingest_samples
Add --reset to wipe the collection first (recommended before the eval, so
leftover documents from earlier runs or admin uploads can't skew results):
    python -m scripts.ingest_samples --reset

Access policy for the demo:
  - hr_policy.txt            -> hr only            (restricted)
  - finance_q3_report.txt    -> finance only       (confidential)
  - engineering_architecture -> engineering + general (internal)
(admin always sees everything.)
"""

import argparse
import sys
from pathlib import Path

# Allow running as a plain script too (python scripts/ingest_samples.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SAMPLE_DOCS_DIR  # noqa: E402
from ingestion.embedder import ingest_file  # noqa: E402
from retrieval.vector_store import reset_collection  # noqa: E402

SAMPLE_POLICY = [
    ("hr_policy.txt", "hr", "restricted", ["hr"]),
    ("finance_q3_report.txt", "finance", "confidential", ["finance"]),
    ("engineering_architecture.txt", "engineering", "internal",
     ["engineering", "general"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the sample documents.")
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the collection before ingesting (clean-slate ingest).",
    )
    args = parser.parse_args()

    if args.reset:
        reset_collection()
        print("reset: collection cleared before ingest\n")

    for filename, department, sensitivity, allowed_roles in SAMPLE_POLICY:
        path = Path(SAMPLE_DOCS_DIR) / filename
        if not path.exists():
            print(f"!! missing sample doc: {path}")
            continue
        n = ingest_file(str(path), department, sensitivity, allowed_roles)
        print(f"ok  {filename}: {n} chunks  (roles: {', '.join(allowed_roles)} + admin)")
    print("\nSample ingestion complete.")


if __name__ == "__main__":
    main()
