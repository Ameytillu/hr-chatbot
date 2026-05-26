# app_lite.py — minimal, stable Streamlit UI

# ---- path shim (so `src/...` imports work) ----
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- imports ----
import streamlit as st
from src.retrieval.search import hybrid_search
from src.llm.generator import generate_answer

# ---- page config ----
st.set_page_config(
    page_title="Hotel HR Smart Desk",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Custom CSS for modern dark theme styling ----
st.markdown("""
<style>
    /* Main container styling - Dark theme */
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 2rem 2rem 2rem;
        margin-bottom: 3rem;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.5);
        border-radius: 50px;
        padding: 12px 24px;
        color: #a5b4fc;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .hero-badge::before {
        content: "🤖 ";
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        line-height: 1.2;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .hero-title-gradient {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #9ca3af;
        max-width: 900px;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
    }
    
    /* Custom card styling - Dark theme */
    .main-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        margin: 2rem auto;
        max-width: 1000px;
    }
    
    /* Section headings */
    .section-heading {
        color: white !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Input field styling - Dark theme */
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        padding: 18px 24px !important;
        font-size: 1.05rem !important;
        color: white !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #6b7280 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.2) !important;
        background: rgba(15, 23, 42, 1) !important;
    }
    
    /* Button styling - Dark theme with gradient */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 16px 48px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.6) !important;
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    }
    
    /* Radio button styling - Dark theme */
    .stRadio > label {
        font-weight: 600;
        color: white !important;
        font-size: 1rem;
    }
    
    .stRadio > div {
        background: rgba(15, 23, 42, 0.6) !important;
        padding: 20px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        display: flex !important;
        gap: 1rem !important;
    }
    
    .stRadio > div > label {
        color: #e5e7eb !important;
        background: rgba(30, 41, 59, 0.6) !important;
        padding: 12px 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stRadio > div > label:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
    }
    
    .stRadio > div > label > div {
        color: #e5e7eb !important;
    }
    
    /* Answer box styling - Dark theme */
    .answer-box {
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #818cf8;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        color: #e5e7eb;
        line-height: 1.8;
    }
    
    /* Expander styling - Dark theme */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        color: #818cf8 !important;
        padding: 1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Snippet card styling - Dark theme */
    .snippet-card {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .snippet-card:hover {
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .snippet-card h4 {
        color: #818cf8 !important;
        margin-bottom: 1rem;
    }
    
    .snippet-card p {
        color: #9ca3af !important;
    }
    
    .snippet-card code {
        background: rgba(99, 102, 241, 0.2) !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        color: #a5b4fc !important;
    }
    
    /* Success/Error messages - Dark theme */
    .stSuccess, .stError, .stInfo {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 12px !important;
        border-left-width: 4px !important;
        color: #e5e7eb !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(99, 102, 241, 0.2) !important;
        margin: 2rem 0 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar - Dark theme */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    
    /* Form container */
    [data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- Hero Section ----
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">Hotel HR Smart Assistant</div>
    <h1 class="hero-title">
        Intelligent HR Support for Hospitality <br/>
        <span class="hero-title-gradient">with polished hotel policy guidance</span>
    </h1>
    <p class="hero-subtitle">
        Find hotel HR policy answers quickly and professionally. This RAG-based assistant uses your policy corpus to provide accurate guidance and, when unsure, gently recommends contacting HR.
    </p>
</div>
""", unsafe_allow_html=True)

# ---- Main content area ----
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    # ---- Q&A form ----
    with st.form("qa", clear_on_submit=False):
        st.markdown('<h3 class="section-heading">💬 Ask Your Question</h3>', unsafe_allow_html=True)
        q = st.text_input(
            "Type your HR policy question here...",
            placeholder="e.g., What is the PTO carryover rule?",
            label_visibility="collapsed"
        )
        
        st.markdown('<h3 class="section-heading">🎨 Response Style</h3>', unsafe_allow_html=True)
        style_label = st.radio(
            "Choose your preferred answer format:",
            ["📝 Focused bullets", "📄 Friendly summary"],
            index=1,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("🔍 Get Answer", use_container_width=True)

    style = {"📝 Focused bullets": "bullets", "📄 Friendly summary": "paragraph"}[style_label]

    if submitted and q.strip():
        with st.spinner('🔍 Searching and generating answer...'):
            try:
                hits = hybrid_search(q.strip())
                st.session_state["last_hits"] = hits
                
                # Display answer in a styled box
                st.markdown("---")
                st.markdown('<h3 class="section-heading">✨ Answer</h3>', unsafe_allow_html=True)
                answer = generate_answer(q, hits, style=style)
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                
                # Success message
                st.success("✅ Answer generated successfully!")
                
            except Exception as e:
                st.error("❌ Error while answering:")
                st.exception(e)

    # ---- Retrieved snippets section ----
    if st.session_state.get("last_hits"):
        st.markdown("---")
        with st.expander("🔎 View Retrieved Snippets & Sources", expanded=False):
            st.markdown(f"**Found {len(st.session_state.get('last_hits', []))} relevant snippets**")
            
            for i, h in enumerate(st.session_state.get("last_hits", []), 1):
                snippet = h.get("text", "")
                if len(snippet) > 500:
                    snippet = snippet[:500] + "…"
                
                # Create styled snippet cards
                st.markdown(f"""
                <div class="snippet-card">
                    <h4>📄 {i}. {h.get('source', 'Unknown Source')}</h4>
                    <p style="color: #666; font-size: 0.9rem;">
                        <strong>Relevance Score:</strong> <code>{h.get('score', 0):.2f}</code> • 
                        <strong>Effective From:</strong> <code>{h.get('effective_from', 'n/a')}</code>
                    </p>
                    <p style="margin-top: 1rem; line-height: 1.6;">{snippet}</p>
                </div>
                """, unsafe_allow_html=True)

# ---- Feature badges below hero ----
st.markdown("""
<div style="text-align: center; margin: 3rem 0;">
    <style>
        .feature-badge {
            display: inline-block;
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 50px;
            padding: 12px 24px;
            margin: 0.5rem;
            color: #e5e7eb;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }
        .feature-badge:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: rgba(99, 102, 241, 0.5);
            transform: translateY(-2px);
        }
    </style>
    <div class="feature-badge">🕐 24/7 Availability</div>
    <div class="feature-badge">📚 RAG Knowledge Base</div>
    <div class="feature-badge">📅 Smart Responses</div>
    <div class="feature-badge">👤 Human-Like Assistance</div>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar with info ----
with st.sidebar:
    st.markdown("### 📚 About Hotel HR Smart Desk")
    st.markdown("""
    This HR assistant helps hotel employees find policy answers from the internal knowledge base.
    
    **Features:**
    - 🔍 Pure RAG search with local policy retrieval
    - 📊 Clear citations from hotel HR documents
    - 🧭 Polite guidance when policies are unclear
    - ⚡ Fast and reliable
    
    **Tips:**
    - Ask in natural language
    - Use specific terms like 'PTO', 'dress code', or 'leave request'
    - If you still need help, contact your HR team
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Examples")
    example_questions = [
        "What is the PTO policy?",
        "How do I request time off?",
        "What is the dress code?",
        "Tell me about health insurance",
        "What are the working hours?"
    ]
    
    for eq in example_questions:
        if st.button(f"💡 {eq}", key=eq, use_container_width=True):
            st.session_state["example_q"] = eq
            st.rerun()

# Handle example question clicks
if "example_q" in st.session_state:
    st.info(f"Try asking: {st.session_state['example_q']}")
    del st.session_state["example_q"]
