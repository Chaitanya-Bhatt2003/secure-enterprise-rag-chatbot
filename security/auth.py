"""Role-based authentication (mock implementation).

Validates username/password against the hashed mock user database in
config.py and stores the authenticated identity in st.session_state.
Session state is server-side in Streamlit, so the role can't be tampered
with from the browser.

This is deliberately simple — in production you'd swap `authenticate()`
for an SSO/OIDC integration and keep the rest of the app unchanged, since
everything downstream only consumes `get_current_user()`.
"""

import hashlib
import hmac
from typing import Optional

import streamlit as st

from config import MOCK_USERS


def authenticate(username: str, password: str) -> Optional[str]:
    """Return the user's role if credentials are valid, else None.

    Uses hmac.compare_digest for constant-time comparison (avoids timing
    side-channels, and is simply the correct habit for credential checks).
    """
    user = MOCK_USERS.get(username)
    if user is None:
        return None
    supplied_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if hmac.compare_digest(supplied_hash, user["password_hash"]):
        return user["role"]
    return None


def login(username: str, password: str) -> bool:
    """Attempt login; on success store identity in session state."""
    role = authenticate(username, password)
    if role is None:
        return False
    st.session_state["user"] = username
    st.session_state["role"] = role
    return True


def logout() -> None:
    """Clear the session identity (and chat history)."""
    for key in ("user", "role", "messages"):
        st.session_state.pop(key, None)


def get_current_user() -> Optional[dict]:
    """The logged-in identity, or None. Single source of truth for RBAC."""
    if "user" in st.session_state and "role" in st.session_state:
        return {"user": st.session_state["user"], "role": st.session_state["role"]}
    return None
