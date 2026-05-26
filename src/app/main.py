# ---------------------- PATH SHIM (must be first!) ----------------------
# Ensures "src/..." imports work when running `streamlit run src/app/main.py`
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # project root (folder with README.md)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------------

import os
import textwrap
import streamlit as st

# Local modules (now resolvable thanks to the shim above)
from src.core.config import settings
from src.retrieval.search import hybrid_search
from src.llm.generator import generate_answer


# ---------------------------- Helpers -----------------------------------
def _index_ready() -> bool:
    """Check if the dense/BM25 index files exist."""
    idx_dir = ROOT / "data" / "index"
    return (idx_dir / "vectors.npy").exists() and (idx_dir / "meta.jsonl").exists()

def _processed_ready() -> bool:
    """Check if processed corpus exists (from ingestion)."""
    return (ROOT / "data" / "processed" / "corpus.jsonl").exists()

def _shorten(s: str, n: int = 180) -> str:
    s = s.strip()
    return (s[: n - 1] + "…") if len(s) > n else s


# ----------------------------- UI ---------------------------------------
st.set_page_config(page_title="Hotel HR Smart Desk", layout="wide")
st.title("Hotel HR Smart Desk")

# Sidebar status
with st.sidebar:
    st.subheader("Status")
    embed_model = getattr(settings, "EMBEDDINGS_MODEL", "text-embedding-3-small")
    embed_provider = getattr(settings, "EMBEDDINGS_PROVIDER", "st")

    st.write("**Embeddings**:", f"`{embed_provider}` · `{embed_model}`")
    st.write("**Mode**:", "`RAG only` — no external OpenAI API key required")

    if getattr(settings, "DATABASE_URL", None):
        st.write("**DB:** configured")
    else:
        st.write("**DB:** not configured *(file-based mode)*")

    # Index/corpus checks
    if not _processed_ready():
        st.warning("No processed corpus found. Run ingestion first:\n\n"
                   "```bash\npython -m src.data_pipeline.cli_ingest --in data/raw_policies --out data/processed\n```")
    elif not _index_ready():
        st.warning("Index not built. Build it with:\n\n"
                   "```bash\npython -m src.retrieval.index_faiss\n```")
    else:
        st.success("Index loaded")

    # Utilities
    if st.button("🧹 Clear chat"):
        st.session_state.pop("history", None)
        st.session_state.pop("last_hits", None)
        st.experimental_rerun()

st.caption("Ask about your hotel HR policies. The bot answers from retrieved snippets with citations and politely suggests contacting HR when a clear match is not found.")


# --------------------------- Chat state ----------------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {"role": "user"|"assistant", "content": str}
if "last_hits" not in st.session_state:
    st.session_state.last_hits = [] # store last retrieval results for the debug panel


# --------------------------- Chat input ----------------------------------
q = st.chat_input("Type your question, e.g., “What is the PTO carryover policy?”")

if q:
    # Guard: require index
    if not _index_ready():
        st.error("No index found. Please run ingestion and build the index (see sidebar).")
    else:
        # Save user message
        st.session_state.history.append({"role": "user", "content": q})

        # Retrieve + answer
        hits = hybrid_search(q)
        st.session_state.last_hits = hits
        answer_md = generate_answer(q, hits)
        st.session_state.history.append({"role": "assistant", "content": answer_md})

# ------------------------ Render conversation ----------------------------
for msg in st.session_state.history[-20:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------ Debug / Evidence panel -------------------------
with st.expander("🔎 View retrieved snippets (debug)"):
    if not st.session_state.last_hits:
        st.info("No snippets yet. Ask a question above.")
    else:
        for i, h in enumerate(st.session_state.last_hits, start=1):
            st.markdown(
                f"**{i}. {h.get('source','(source)')}**  \n"
                f"*score:* `{h.get('score', 0):.2f}` • "
                f"*effective:* `{h.get('effective_from','n/a')}`  \n"
                f"{_shorten(h.get('text',''))}"
            )

# ------------------------ Empty-state guidance ---------------------------
if not st.session_state.history:
    if _index_ready():
        st.success("✅ Index is ready. Type a question, e.g.: "
                   "“What’s the PTO carryover rule?” or “What does the dress code say?”")
    elif _processed_ready():
        st.info(
            "Processed corpus found but no index yet.\n\n"
            "Build it with:\n\n"
            "```bash\npython -m src.retrieval.index_faiss\n```"
        )
    else:
        st.info(
            "Tip: put your Markdown policies in `data/raw_policies/`, then run:\n\n"
            "```bash\n"
            "python -m src.data_pipeline.cli_ingest --in data/raw_policies --out data/processed\n"
            "python -m src.retrieval.index_faiss\n"
            "```"
        )
