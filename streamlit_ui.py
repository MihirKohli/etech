"""
Streamlit UI — chat interface with session management and document upload.

Run:  streamlit run app/ui/streamlit_app.py
"""

import streamlit as st
import requests
from config import get_settings

API_URL = get_settings().API_URL

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("📚 Conversational RAG System")


# ── Helpers ──────────────────────────────────────────

def api_get(path, params=None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path, json=None, files=None, params=None):
    try:
        r = requests.post(f"{API_URL}{path}", json=json, files=files, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Sidebar: User + Sessions + Upload ────────────────

with st.sidebar:
    st.header("Settings")

    user_id = st.text_input("User ID", value="default_user")

    st.divider()

    # Session management
    st.subheader("Sessions")

    if st.button("➕ New Session"):
        result = api_post("/sessions", params={"user_id": user_id})
        if result:
            st.session_state["session_id"] = result["session_id"]
            st.session_state["messages"] = []
            st.rerun()

    # List existing sessions
    sessions = api_get("/sessions", params={"user_id": user_id})
    if sessions:
        session_options = {
            f"{s['preview']} ({s['turn_count']} turns)": s["session_id"]
            for s in sessions
        }
        if session_options:
            selected = st.selectbox("Pick a session", options=list(session_options.keys()))
            if st.button("Load"):
                sid = session_options[selected]
                st.session_state["session_id"] = sid
                history = api_get(f"/sessions/{sid}/messages")
                st.session_state["messages"] = history or []
                st.rerun()

    st.divider()

    # Document upload
    st.subheader("Upload Documents")
    uploaded = st.file_uploader(
        "PDF, Markdown, or HTML",
        type=["pdf", "md", "html"],
    )
    if uploaded and st.button("Upload & Ingest"):
        if "session_id" not in st.session_state:
            st.error("Create or load a session before uploading documents.")
        else:
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            result = api_post("/documents/upload", files=files, params={"session_id": st.session_state["session_id"]})
            if result:
                st.success(f"✓ {result['filename']} → {result['chunks_created']} chunks")

    st.divider()

    # Session info
    if "session_id" in st.session_state:
        info = api_get(f"/sessions/{st.session_state['session_id']}")
        if info:
            st.caption(f"Session: `{info['session_id'][:8]}...`")
            st.caption(f"Turns: {info['turn_count']}")
            if info.get("summary"):
                with st.expander("Summary"):
                    st.write(info["summary"])


# ── Main Chat Area ───────────────────────────────────

if "session_id" not in st.session_state:
    st.info("Create or load a session from the sidebar to start chatting.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.caption(f"**{s['document_name']}** (score: {s['score']:.2f})")
                    st.text(s["snippet"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = api_post("/chat", json={
                "session_id": st.session_state["session_id"],
                "message": prompt,
            })

        if result:
            st.write(result["answer"])

            sources = result.get("sources", [])
            if sources:
                with st.expander("📄 Sources"):
                    for s in sources:
                        st.caption(f"**{s['document_name']}** (score: {s['score']:.2f})")
                        st.text(s["snippet"])

            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": sources,
            })
        else:
            st.error("Failed to get response.")