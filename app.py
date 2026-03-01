"""
Glass Expert AI — app.py
Streamlit Chat Interface for the RAG System

Run with:
    streamlit run app.py
"""
import os
import sys
import time
import httpx
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glass Expert AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080/api/v1")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Chat message styling */
    .user-message {
        background: #1e3a5f;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e8f4fd;
        font-size: 15px;
    }
    .assistant-message {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 2px 12px 12px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e2e8f0;
        font-size: 15px;
    }

    /* Source card */
    .source-card {
        background: #1a2332;
        border: 1px solid #2d4a6e;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #94a3b8;
    }
    .source-card strong { color: #60a5fa; }
    .similarity-bar {
        height: 4px;
        background: linear-gradient(to right, #3b82f6, #06b6d4);
        border-radius: 2px;
        margin-top: 6px;
    }

    /* Metrics */
    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: bold;
        color: #60a5fa;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1a1f2e;
        color: #e2e8f0;
        border: 1px solid #2d3748;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "avg_response_time" not in st.session_state:
    st.session_state.avg_response_time = 0.0


# ── Helper Functions ───────────────────────────────────────────────────────────
def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception:
        return None


def query_rag(question: str, top_k: int, source_type: str = None, language: str = None) -> dict:
    """Send a query to the RAG API and return the response."""
    payload = {"question": question, "top_k": top_k}
    if source_type and source_type != "All":
        payload["source_type"] = source_type.lower()
    if language and language != "Auto-detect":
        payload["language"] = "fa" if language == "Farsi" else "en"

    try:
        r = httpx.post(f"{API_BASE}/query", json=payload, timeout=120)
        return r.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to API server. Make sure it is running."}
    except httpx.TimeoutException:
        return {"error": "Request timed out. The LLM may be loading — try again in 30 seconds."}
    except Exception as e:
        return {"error": str(e)}


def render_source_card(source: dict, index: int):
    """Render a source citation card."""
    similarity_pct = int(source["similarity"] * 100)
    bar_width = similarity_pct
    lang_flag = "🇮🇷" if source["language"] == "fa" else "🇬🇧"

    st.markdown(f"""
    <div class="source-card">
        <strong>📄 {source['title']}</strong>
        &nbsp;&nbsp;
        <span style="background:#1e3a5f; padding:2px 8px; border-radius:4px; font-size:11px;">
            {source['source_type']}
        </span>
        &nbsp;{lang_flag}&nbsp;
        <span style="color:#34d399; font-size:12px;">
            {similarity_pct}% match
        </span>
        <div class="similarity-bar" style="width:{bar_width}%;"></div>
        <div style="margin-top:6px; color:#64748b; font-size:12px; font-style:italic;">
            {source['content_preview'][:200]}...
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Glass Expert AI")
    st.markdown("*Powered by BAAI/bge-m3 + pgvector*")
    st.divider()

    # API Health Check
    health = check_api_health()
    if health:
        db_color = "🟢" if health.get("database") == "healthy" else "🔴"
        redis_color = "🟢" if health.get("redis") == "healthy" else "🟡"
        st.markdown(f"""
        **System Status**
        - {db_color} Database: {health.get('database', 'unknown')}
        - {redis_color} Cache: {health.get('redis', 'unknown')}
        - 🟢 API: online
        - 📚 Chunks: **{health.get('total_chunks', 0):,}**
        - 📄 Documents: **{health.get('total_documents', 0):,}**
        """)
    else:
        st.error("⚠️ API server is offline\n\nRun: `python -m uvicorn api.main:app --port 8080 --reload`")

    st.divider()

    # Query Settings
    st.markdown("**Query Settings**")
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=5)
    source_filter = st.selectbox(
        "Filter by source type",
        ["All", "Textbook", "Paper", "SOP", "Standard", "Manual", "QA Pair"],
    )
    language_mode = st.selectbox(
        "Language",
        ["Auto-detect", "English", "Farsi"],
    )

    st.divider()

    # Session Stats
    st.markdown("**Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{st.session_state.total_queries}</div>
            Queries
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_ms = st.session_state.avg_response_time
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{avg_ms:.0f}</div>
            Avg ms
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.avg_response_time = 0.0
        st.rerun()

    # Ingest new file
    st.markdown("**Add to Knowledge Base**")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "csv", "txt", "docx", "json"],
        help="Upload a glass science document to add to the knowledge base",
    )
    ingest_type = st.selectbox("Document type", ["textbook", "paper", "sop", "standard", "manual"])
    if uploaded_file and st.button("📥 Ingest Document", use_container_width=True):
        # Save to temp location and trigger ingestion
        temp_path = Path("data/uploads") / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(uploaded_file.getvalue())
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                r = httpx.post(
                    f"{API_BASE}/ingest",
                    json={"file_path": str(temp_path.resolve()), "source_type": ingest_type},
                    timeout=300,
                )
                result = r.json()
                if result.get("status") == "success":
                    st.success(f"✅ Ingested {result['chunks_stored']} chunks")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('message', 'Ingestion failed')}")
            except Exception as e:
                st.error(f"❌ Error: {e}")


# ── Main Chat Interface ────────────────────────────────────────────────────────
st.markdown("## 🔬 Glass Expert AI")
st.markdown("Ask any question about glass science, manufacturing, composition, defects, or properties.")

# Render chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🔬 {msg["content"]}</div>', unsafe_allow_html=True)

            # Show sources if available
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} source(s) used — click to view", expanded=False):
                    for i, source in enumerate(msg["sources"]):
                        render_source_card(source, i)

            # Show metadata
            if msg.get("meta"):
                meta = msg["meta"]
                cols = st.columns(4)
                cols[0].caption(f"⏱ {meta.get('retrieval_time_ms', 0):.0f}ms retrieval")
                cols[1].caption(f"🌐 {meta.get('language_detected', 'en').upper()}")
                cols[2].caption(f"🤖 {meta.get('model_used', 'unknown')}")
                cols[3].caption(f"📊 {meta.get('total_chunks_searched', 0)} chunks searched")

# ── Suggested Questions ────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    suggestions = [
        "What is the glass transition temperature of borosilicate glass?",
        "What causes devitrification in glass manufacturing?",
        "How does silica content affect glass viscosity?",
        "What are the corrective actions for bubbles in glass?",
        "Explain the Zachariasen network theory of glass structure.",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state._pending_question = suggestion
                st.rerun()

# ── Chat Input ─────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a glass science question...")

# Handle suggested question clicks
if hasattr(st.session_state, "_pending_question"):
    question = st.session_state._pending_question
    del st.session_state._pending_question

if question:
    if not health:
        st.error("⚠️ The API server is not running. Start it first:\n```\npython -m uvicorn api.main:app --port 8080 --reload\n```")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    # Query the RAG API
    with st.spinner("🔍 Searching knowledge base..."):
        start_time = time.time()
        response = query_rag(
            question=question,
            top_k=top_k,
            source_type=source_filter,
            language=language_mode,
        )
        elapsed_ms = (time.time() - start_time) * 1000

    if "error" in response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Error: {response['error']}",
            "sources": [],
            "meta": {},
        })
    else:
        # Update session stats
        st.session_state.total_queries += 1
        n = st.session_state.total_queries
        prev_avg = st.session_state.avg_response_time
        st.session_state.avg_response_time = (prev_avg * (n - 1) + elapsed_ms) / n

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("answer", "No answer generated."),
            "sources": response.get("sources", []),
            "meta": {
                "retrieval_time_ms": response.get("retrieval_time_ms", 0),
                "language_detected": response.get("language_detected", "en"),
                "model_used": response.get("model_used", "unknown"),
                "total_chunks_searched": response.get("total_chunks_searched", 0),
            },
        })

    st.rerun()
