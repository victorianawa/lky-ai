"""Ask LKY - Streamlit chat UI."""
from __future__ import annotations

import os
import sys

# Streamlit Community Cloud (Linux) sqlite3 fix for ChromaDB
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

# Streamlit Cloud: expose secrets as env vars so chatbot.py has one config path.
try:
    for _k in ("GEMINI_API_KEY", "GEMINI_MODEL", "WEAK_DISTANCE"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file locally; .env is used instead

import chatbot  # noqa: E402
import ingest  # noqa: E402

st.set_page_config(
    page_title="Ask LKY - AI Historical Knowledge Assistant",
    page_icon="🇸🇬",
    layout="centered",
    initial_sidebar_state="expanded",
)

EXAMPLES = [
    "Why did Singapore leave Malaysia in 1965, and what were the immediate challenges?",
    "How did Singapore attract multinational companies and foreign investment?",
    "What core qualities make an effective and resilient leader?",
    "What might he think about artificial intelligence and its impact on the economy?",
]


@st.cache_resource(show_spinner="Initializing knowledge base (loading embedder and ChromaDB)...")
def init_knowledge_base() -> dict:
    """Load the embedder and build the index if missing (e.g. on ephemeral cloud instances)."""
    ingest.get_embedder()
    info = {"warnings": []}
    if ingest.get_collection().count() == 0:
        info = ingest.ingest()
    info["chunks"] = ingest.get_collection().count()
    return info


def render_sources(sources: list, weak: bool) -> None:
    """Render metadata-driven source cards strictly from vector DB results."""
    if not sources:
        return
    if weak:
        st.warning(
            "⚠️ **Weak Evidence Notice:** The retrieved passages have lower similarity scores and "
            "may not directly answer this question. Grounding rules require the model to acknowledge limitations.",
            icon="⚠️",
        )
    with st.expander(f"📚 Retrieved Context Sources ({len(sources)} passages)", expanded=False):
        st.caption(
            "These passages were retrieved by vector search and passed to Gemini. "
            "Sources originate from document metadata, not model generation."
        )
        for i, s in enumerate(sources, 1):
            title = getattr(s, "title", "") or getattr(s, "source", "Document")
            page_info = f", p. {s.page}" if getattr(s, "page", 0) else ""
            category = getattr(s, "category", "general")
            dist = getattr(s, "distance", 0.0)

            st.markdown(f"**[{i}] {title}**{page_info} · *{category}*")
            st.markdown(f"> {s.text}")
            meta_line = f"Cosine Distance: `{dist:.3f}` · Chunk: `{s.chunk_id}`"
            if getattr(s, "url", ""):
                meta_line += f" · [Source Link]({s.url})"
            st.caption(meta_line)
            if i < len(sources):
                st.divider()


# ----------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.markdown("## 🇸🇬 Ask LKY")
    st.caption("AI Historical Knowledge Assistant")
    st.write(
        "A pragmatic, evidence-based simulation inspired by the documented speeches, writings, "
        "and interviews of Lee Kuan Yew (1923–2015)."
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("System Status")
    kb_info = init_knowledge_base()
    active_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

    st.markdown(f"**Model:** `{active_model}`")
    st.markdown(f"**Corpus Chunks:** `{kb_info['chunks']}`")
    st.markdown("**Embedder:** `all-MiniLM-L6-v2`")
    st.markdown("**Database:** `ChromaDB (Cosine)`")

    st.markdown("---")
    st.subheader("Architecture")
    st.markdown(
        "- **Custom RAG:** Direct retrieval & prompt assembly (no LangChain)\n"
        "- **Grounded Citations:** References tied to retrieved passages\n"
        "- **Hypothetical Guardrail:** Post-2015 topics explicitly labeled\n"
        "- **Abstention Support:** Explains when corpus lacks evidence"
    )
    st.caption("Built for the What Would Lee Kuan Yew Do? challenge.")


# -------------------------------------------------------------------- main ----
st.title("🇸🇬 Ask LKY")
st.caption("Explore leadership, history, economic policy, and international relations.")

with st.expander("ℹ️ AI Historical Simulation & Grounding Notice", expanded=False):
    st.markdown(
        "**Important:** This application is an AI historical simulation developed for educational and research "
        "exploration. It is **not** Lee Kuan Yew himself. The model has no personal memories, cannot invent quotes, "
        "is grounded in the retrieved documents, and labels any events after March 2015 as hypothetical extrapolation."
    )

if kb_info["chunks"] == 0:
    st.error("The knowledge base is empty. Please add documents to `data/<category>/` and run `python ingest.py`.")
    for w in kb_info.get("warnings", []):
        st.caption(w)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for m in st.session_state.messages:
    avatar = "👤" if m["role"] == "user" else "🇸🇬"
    with st.chat_message(m["role"], avatar=avatar):
        if m.get("error"):
            st.error(m["content"])
        else:
            st.markdown(m["content"])
        render_sources(m.get("sources", []), m.get("weak", False))

# Handle example click from session state
user_question = None
if "example_choice" in st.session_state:
    user_question = st.session_state.pop("example_choice")

# Prompt buttons if conversation is empty
if not st.session_state.messages and not user_question:
    st.markdown("##### Suggested Questions")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.example_choice = ex
            st.rerun()

# User input field
chat_input = st.chat_input("Ask about Singapore history, economic policy, geopolitics, or leadership...")
if chat_input:
    user_question = chat_input

if user_question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages if not m.get("error")
    ]
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar="🇸🇬"):
        with st.spinner("Searching historical documents..."):
            res = chatbot.ask_stream(user_question, history)

        if res.error:
            st.error(res.error)
            full_response = res.error
            is_error = True
        else:
            full_response = st.write_stream(res.stream)
            is_error = False

        render_sources(res.sources, res.weak_evidence)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": res.sources,
        "weak": res.weak_evidence,
        "error": is_error,
    })
