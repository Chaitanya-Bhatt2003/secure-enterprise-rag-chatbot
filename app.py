"""Secure Enterprise RAG Chatbot — Streamlit entry point.

Run with:  streamlit run app.py

Pages:
  - Login (always first; nothing is reachable without a session role)
  - Chat (all authenticated roles)
  - Document Upload (admin only — enforced server-side, not just hidden)
"""

from pathlib import Path

import streamlit as st

import config
from security.auth import get_current_user, login, logout

st.set_page_config(
    page_title="Secure Enterprise RAG",
    page_icon="🔐",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Login screen
# ---------------------------------------------------------------------------
def render_login() -> None:
    st.title("🔐 Secure Enterprise RAG Chatbot")
    st.caption("Sign in to access the document assistant. Access is role-based.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        if login(username.strip(), password):
            st.rerun()
        else:
            st.error("Invalid username or password.")

    with st.expander("Demo accounts"):
        st.markdown(
            "| Username | Password | Role |\n"
            "|---|---|---|\n"
            "| admin | admin123 | admin |\n"
            "| hannah | hr123 | hr |\n"
            "| frank | fin123 | finance |\n"
            "| erin | eng123 | engineering |\n"
            "| guest | guest123 | general |"
        )


# ---------------------------------------------------------------------------
# Sidebar (identity + session controls)
# ---------------------------------------------------------------------------
def render_sidebar(identity: dict) -> str:
    with st.sidebar:
        st.markdown("### 👤 Session")
        st.markdown(f"**User:** `{identity['user']}`")
        st.markdown(f"**Role:** `{identity['role']}`")

        try:
            from retrieval.vector_store import collection_stats
            st.markdown(f"**Indexed chunks:** {collection_stats()['chunks']}")
        except Exception:
            pass

        st.divider()
        pages = ["💬 Chat"]
        if identity["role"] == "admin":
            pages.append("📤 Document Upload")
        page = st.radio("Page", pages, label_visibility="collapsed")

        st.divider()
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()
    return page


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------
def render_chat(identity: dict) -> None:
    st.title("💬 Document Assistant")
    st.caption(
        f"Answers are grounded in documents your role (`{identity['role']}`) "
        "is permitted to access. Sensitive data is masked per policy."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Replay history (sources expander included for past answers).
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"], msg.get("masked_entities", []))

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering…"):
            try:
                from llm.chain import answer_query
                result = answer_query(question, identity["user"], identity["role"])
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                return

        st.markdown(result.answer)
        if result.injection_flagged:
            st.warning("This query was blocked by the prompt-injection guardrail.")
        if result.output_redacted:
            st.info("Some content was redacted from this answer by the output filter.")

        sources = [
            {
                "source": c.source,
                "page": c.page,
                "department": c.metadata.get("department", "?"),
                "sensitivity": c.metadata.get("sensitivity_level", "?"),
                "text": c.text,
            }
            for c in result.sources
        ]
        if sources:
            _render_sources(sources, result.masked_entities)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": sources if not result.injection_flagged else [],
            "masked_entities": result.masked_entities,
        }
    )


def _render_sources(sources: list, masked_entities: list) -> None:
    """Sources expander: which (already-masked) chunks fed the answer."""
    with st.expander(f"📎 Sources ({len(sources)})"):
        if masked_entities:
            st.caption(f"🔒 Masked for your role: {', '.join(masked_entities)}")
        for i, src in enumerate(sources, 1):
            st.markdown(
                f"**{i}. {src['source']}** — page {src['page']} · "
                f"dept: `{src['department']}` · sensitivity: `{src['sensitivity']}`"
            )
            st.text(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""))
            st.divider()


# ---------------------------------------------------------------------------
# Admin upload page
# ---------------------------------------------------------------------------
def render_upload(identity: dict) -> None:
    # Server-side enforcement: the sidebar hides this page from non-admins,
    # but never TRUST the UI — re-check the session role here too.
    if identity["role"] != "admin":
        st.error("You do not have permission to upload documents.")
        return

    st.title("📤 Document Upload")
    st.caption("Upload a document, tag its access policy, and ingest it into the index.")

    uploaded = st.file_uploader(
        "Choose a file", type=["pdf", "docx", "txt", "csv", "md"]
    )
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Department", config.DEPARTMENTS)
        sensitivity = st.selectbox("Sensitivity level", config.SENSITIVITY_LEVELS, index=1)
    with col2:
        allowed_roles = st.multiselect(
            "Allowed roles (admin always included)",
            [r for r in config.ROLES if r != "admin"],
            default=["general"],
        )

    if uploaded and st.button("Ingest document", type="primary"):
        uploads_dir = Path(config.PROJECT_ROOT) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        dest = uploads_dir / uploaded.name
        dest.write_bytes(uploaded.getbuffer())

        with st.spinner("Loading → chunking → tagging → embedding…"):
            try:
                from ingestion.embedder import ingest_file
                from utils.logger import audit_log

                n = ingest_file(str(dest), department, sensitivity, allowed_roles)
                audit_log(
                    identity["user"], identity["role"],
                    f"uploaded {uploaded.name}",
                    sources=[{"source_file": uploaded.name,
                              "department": department,
                              "sensitivity_level": sensitivity,
                              "allowed_roles": allowed_roles}],
                    event="ingest",
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                return
        st.success(
            f"Ingested **{uploaded.name}**: {n} chunks stored. "
            f"Visible to roles: {', '.join(sorted(set(allowed_roles) | {'admin'}))}."
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main() -> None:
    identity = get_current_user()
    if identity is None:
        render_login()
        return

    page = render_sidebar(identity)
    if page == "📤 Document Upload":
        render_upload(identity)
    else:
        render_chat(identity)


if __name__ == "__main__":
    main()
