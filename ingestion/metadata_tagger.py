"""Metadata tagging — the foundation of RBAC in this system.

Every chunk gets:
  - department          : which department owns the document
  - sensitivity_level   : public / internal / confidential / restricted
  - allowed_roles       : comma-separated list (human-readable audit field)
  - role_<role>         : one BOOLEAN per role (the actual filter field)
  - source_file, page, upload_date

SECURITY NOTE — why the boolean flags:
ChromaDB metadata values must be scalars (str/int/float/bool); it cannot
store lists and cannot do substring matching in `where` filters. So instead
of filtering on a list of allowed roles at query time, we "explode" the
allowed_roles list into one boolean flag per role at INGESTION time.
Retrieval then filters with `where={"role_<current_role>": True}` inside
the Chroma query itself — restricted chunks are excluded BEFORE similarity
ranking, never post-filtered in Python.

Admin is always granted access to every chunk (role_admin=True), which is
what lets admins see unmasked, unrestricted results.
"""

from datetime import date
from typing import List

from langchain_core.documents import Document

from config import ROLES, SENSITIVITY_LEVELS


def tag_chunks(
    chunks: List[Document],
    department: str,
    sensitivity_level: str,
    allowed_roles: List[str],
    source_file: str,
) -> List[Document]:
    """Attach RBAC + provenance metadata to every chunk (in place).

    `allowed_roles` is validated against the known role list so a typo can
    never silently create an unreachable (or worse, over-shared) document.
    """
    if sensitivity_level not in SENSITIVITY_LEVELS:
        raise ValueError(f"Unknown sensitivity level: {sensitivity_level}")

    invalid = [r for r in allowed_roles if r not in ROLES]
    if invalid:
        raise ValueError(f"Unknown role(s) in allowed_roles: {invalid}")

    # Admin can always access everything; force-include it so an admin's
    # retrieval filter (role_admin=True) matches every chunk.
    effective_roles = sorted(set(allowed_roles) | {"admin"})

    for chunk in chunks:
        chunk.metadata.update(
            {
                "department": department,
                "sensitivity_level": sensitivity_level,
                "allowed_roles": ",".join(effective_roles),
                "source_file": source_file,
                "upload_date": date.today().isoformat(),
                # `page` may already exist (PDFs); default for other types.
                "page": chunk.metadata.get("page", 0),
            }
        )
        # Explode roles into boolean flags — the fields RBAC filtering
        # actually runs on (see module docstring).
        for role in ROLES:
            chunk.metadata[f"role_{role}"] = role in effective_roles

    return chunks
