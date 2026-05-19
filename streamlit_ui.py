"""
Streamlit UI — chat interface with session management and document upload.

Run:  streamlit run app/ui/streamlit_app.py
"""

import json
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
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # Use a dict so the nested generator can mutate shared state
        # (nonlocal doesn't work at module scope)
        state = {"full_answer": "", "sources": [], "error": False}

        def token_stream():
            """Generator that yields tokens from the SSE stream."""
            try:
                with requests.post(
                    f"{API_URL}/chat/stream",
                    json={"session_id": st.session_state["session_id"], "message": prompt},
                    stream=True,
                    timeout=60,
                ) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                        if not line.startswith("data: "):
                            continue
                        payload = line[len("data: "):]
                        if payload.startswith("[DONE]"):
                            done_data = json.loads(payload[len("[DONE] "):])
                            state["sources"] = done_data.get("sources", [])
                        else:
                            state["full_answer"] += payload
                            yield payload
            except Exception as e:
                state["error"] = True
                st.error(f"Stream error: {e}")

        st.write_stream(token_stream())

        if not state["error"]:
            if state["sources"]:
                with st.expander("📄 Sources"):
                    for s in state["sources"]:
                        st.caption(f"**{s['document_name']}** (score: {s['score']:.2f})")
                        st.text(s["snippet"])

            st.session_state["messages"].append({
                "role": "assistant",
                "content": state["full_answer"],
                "sources": state["sources"],
            })


# ── Explainability Dashboard ──────────────────────────

st.divider()
with st.expander("🔍 Agent Decision Log (Explainability Dashboard)"):
    if "session_id" in st.session_state:
        traces = api_get(f"/chat/trace/{st.session_state['session_id']}")
        if traces:
            for t in traces:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Turn {t['turn']}**")
                    st.caption(f"Intent: `{t['query_intent']}`")
                    st.caption(f"Strategy: `{t['retrieval_strategy']}`")
                with col2:
                    if t.get("rewritten_query"):
                        st.markdown(f"Rewritten: _{t['rewritten_query']}_")
                    if t.get("sub_questions"):
                        st.markdown("Sub-questions:")
                        for sq in t["sub_questions"]:
                            st.markdown(f"- {sq}")
                st.divider()
        else:
            st.info("No agent traces yet. Send a message to see the decision log.")
    else:
        st.info("Load a session to view the agent decision log.")