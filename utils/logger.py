"""Audit logging.

Every query is appended as one JSON line to logs/audit.log: who asked,
with what role, what they asked, which sources were retrieved, what got
masked, and whether guardrails flagged anything. JSON-lines format so the
trail is trivially machine-parseable for compliance review.

The audit trail records source file names and metadata — never chunk
text — so the log itself can't become a PII leak.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import AUDIT_LOG_FILE

_logger: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    """File logger singleton; creates logs/ on first use."""
    global _logger
    if _logger is None:
        Path(AUDIT_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        _logger = logging.getLogger("audit")
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            handler = logging.FileHandler(AUDIT_LOG_FILE, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            _logger.addHandler(handler)
        _logger.propagate = False
    return _logger


def audit_log(
    user: str,
    role: str,
    question: str,
    sources: List[dict],
    masked_entities: Optional[List[str]] = None,
    injection_flagged: bool = False,
    event: str = "query",
) -> None:
    """Append one audit record. Never raises — auditing must not take the
    app down, though failures are printed so they aren't silent."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "user": user,
        "role": role,
        "question": question,
        "sources": sources,
        "masked_entities": masked_entities or [],
        "injection_flagged": injection_flagged,
    }
    try:
        _get_logger().info(json.dumps(record, ensure_ascii=False))
    except Exception as exc:  # pragma: no cover
        print(f"[audit] failed to write audit record: {exc}")
