import streamlit as st
import sqlite3
import bcrypt
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════
def init_db():
    con = sqlite3.connect("users.db")
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY,
        username TEXT,
        role TEXT,
        content TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    con.commit()
    con.close()

def register_user(username, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        con = sqlite3.connect("users.db")
        con.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        con.commit()
        con.close()
        return True
    except:
        return False

def verify_user(username, password):
    con = sqlite3.connect("users.db")
    row = con.execute("SELECT password FROM users WHERE username=?", (username,)).fetchone()
    con.close()
    if row:
        return bcrypt.checkpw(password.encode(), row[0])
    return False

def save_message(username, role, content):
    con = sqlite3.connect("users.db")
    con.execute("INSERT INTO messages (username, role, content) VALUES (?, ?, ?)", (username, role, content))
    con.commit()
    con.close()

def load_messages(username):
    con = sqlite3.connect("users.db")
    rows = con.execute(
        "SELECT role, content FROM messages WHERE username=? ORDER BY id",
        (username,)
    ).fetchall()
    con.close()
    return [{"role": r, "content": c} for r, c in rows]

def clear_messages(username):
    con = sqlite3.connect("users.db")
    con.execute("DELETE FROM messages WHERE username=?", (username,))
    con.commit()
    con.close()

# ════════════════════════════════════════════
# MODEL (cached)
# ════════════════════════════════════════════
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return tokenizer, model

def generate_answer(prompt):
    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ════════════════════════════════════════════
# RAG — PDF PROCESSING
# ════════════════════════════════════════════
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(
    docs,
    get_embeddings()
)
retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)
    return retriever, len(docs)

def get_answer(question, retriever):
    retrieved_docs = retriever.invoke(question)
    seen = set()
    unique_chunks = []
    for doc in retrieved_docs:
        text = doc.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_chunks.append(text)
    context = "\n\n".join(unique_chunks)
    prompt = f"""Answer the question based on the following text.

Text:
{context}

Question: {question}
Answer:"""
    answer = generate_answer(prompt)
    return answer, unique_chunks

# ════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    /* ── Page background ── */
    .stApp { background-color: #0f1117; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 860px;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 8px 14px;
        margin-bottom: 8px;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #c9d1d9 !important;
        font-size: 14px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #0B6477 !important;
        box-shadow: 0 0 0 2px rgba(11,100,119,0.3) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        border: 1px solid #30363d !important;
        background: #21262d !important;
        color: #c9d1d9 !important;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #0B6477 !important;
        border-color: #0B6477 !important;
        color: #ffffff !important;
    }

    /* ── Primary button (Login/Register) ── */
    .stButton > button[kind="primary"] {
        background: #0B6477 !important;
        border-color: #0B6477 !important;
        color: #ffffff !important;
    }

    /* ── Text inputs ── */
    .stTextInput > div > input, .stTextInput > div > div > input {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
        font-size: 14px !important;
    }
    .stTextInput > div > input:focus {
        border-color: #0B6477 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 10px;
        gap: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b949e;
        font-size: 14px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #0B6477 !important;
        color: #ffffff !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px 14px;
    }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 22px !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px !important; }

    /* ── Success / Info / Warning ── */
    .stSuccess { background: #0d2b1e !important; border: 1px solid #1a7a43 !important; border-radius: 8px; }
    .stInfo    { background: #0d1f33 !important; border: 1px solid #1a4a7a !important; border-radius: 8px; }
    .stWarning { background: #2b1d0d !important; border: 1px solid #7a4a1a !important; border-radius: 8px; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #161b22;
        border: 1px dashed #30363d;
        border-radius: 10px;
        padding: 8px;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
        color: #8b949e !important;
        font-size: 13px !important;
    }
    .streamlit-expanderContent {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* ── Divider ── */
    hr { border-color: #21262d !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    /* ── Auth card ── */
    .auth-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 16px;
        padding: 32px;
        max-width: 440px;
        margin: 60px auto 0;
    }
    .auth-title {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 6px;
    }
    .auth-sub {
        font-size: 14px;
        color: #8b949e;
        text-align: center;
        margin-bottom: 28px;
    }

    /* ── App header ── */
    .app-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .app-title {
        font-size: 22px;
        font-weight: 700;
        color: #58a6ff;
        margin: 0;
    }
    .app-subtitle {
        font-size: 13px;
        color: #8b949e;
        margin: 0;
    }

    /* ── Chunk card ── */
    .chunk-card {
        background: #0d1f33;
        border: 1px solid #1a4a7a;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #c9d1d9;
        line-height: 1.6;
    }
    .chunk-label {
        font-size: 11px;
        font-weight: 600;
        color: #58a6ff;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    /* ── Welcome banner ── */
    .welcome-banner {
        background: linear-gradient(135deg, #0d1f33, #0d2b1e);
        border: 1px solid #1a4a7a;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        text-align: center;
    }
    .welcome-title {
        font-size: 20px;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 6px;
    }
    .welcome-sub {
        font-size: 14px;
        color: #8b949e;
    }
    </style>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# AUTH UI
# ════════════════════════════════════════════
def show_auth():
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">📄 DocChat</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Your AI-powered PDF assistant</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Login", "✨ Register"])

    with tab1:
        username = st.text_input("Username", key="login_user", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Login →", key="login_btn", use_container_width=True, type="primary"):
            if not username or not password:
                st.error("Please fill in all fields.")
            elif verify_user(username, password):
                st.session_state["user"] = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

    with tab2:
        new_user = st.text_input("Choose Username", key="reg_user", placeholder="e.g. shreya123")
        new_pass = st.text_input("Choose Password", type="password", key="reg_pass", placeholder="Min 6 characters")
        confirm  = st.text_input("Confirm Password", type="password", key="reg_confirm", placeholder="Repeat password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create Account →", key="reg_btn", use_container_width=True, type="primary"):
            if not new_user or not new_pass or not confirm:
                st.error("Please fill in all fields.")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_pass != confirm:
                st.error("Passwords do not match.")
            elif register_user(new_user, new_pass):
                st.success("✅ Account created! Please login.")
            else:
                st.error("Username already taken. Try another.")

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
def show_sidebar():
    with st.sidebar:
        st.markdown("## 📄 DocChat")
        st.markdown(f"👤 **{st.session_state['user']}**")
        st.divider()

        # ── PDF Upload ──
        st.markdown("### Upload PDF")
        uploaded_file = st.file_uploader(
            label="Drop PDF here",
            type="pdf",
            label_visibility="collapsed"
        )

        if uploaded_file:
            with st.spinner("Processing PDF..."):
                retriever, chunk_count = process_pdf(uploaded_file)
                st.session_state["retriever"] = retriever
                st.session_state["chunk_count"] = chunk_count
                st.session_state["pdf_name"] = uploaded_file.name
                st.session_state["pdf_ready"] = True

        if st.session_state.get("pdf_ready"):
            st.success(f"✅ {st.session_state.get('pdf_name', 'PDF')} ready")
            col1, col2 = st.columns(2)
            col1.metric("Chunks", st.session_state.get("chunk_count", 0))
            col2.metric("Retrieved", 3)

        st.divider()

        # ── Model info ──
        st.markdown("### Model Info")
        st.caption("🤖 flan-t5-base")
        st.caption("🔍 MiniLM-L6-v2")
        st.caption("🗄️ ChromaDB (MMR)")

        st.divider()

        # ── Actions ──
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_messages(st.session_state["user"])
            st.session_state["messages"] = []
            st.rerun()

        if st.button("🚪 Logout", use_container_width=True):
            for key in ["user", "messages", "pdf_ready", "retriever", "chunk_count", "pdf_name"]:
                st.session_state.pop(key, None)
            st.rerun()

    return uploaded_file

# ════════════════════════════════════════════
# MAIN CHAT UI
# ════════════════════════════════════════════
def show_chat():
    # Header
    st.markdown("""
    <div class="app-header">
        <span style="font-size:28px">📄</span>
        <div>
            <p class="app-title">DocChat</p>
            <p class="app-subtitle">Ask anything from your PDF</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Welcome or PDF status banner
    if not st.session_state.get("pdf_ready"):
        st.markdown("""
        <div class="welcome-banner">
            <div class="welcome-title">👋 Welcome to DocChat</div>
            <div class="welcome-sub">Upload a PDF from the sidebar to get started.<br>
            Then ask any question and get instant AI-powered answers.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pdf_name = st.session_state.get("pdf_name", "your PDF")
        chunks   = st.session_state.get("chunk_count", 0)
        st.markdown(f"""
        <div style="background:#0d2b1e;border:1px solid #1a7a43;border-radius:10px;
                    padding:10px 18px;margin-bottom:16px;font-size:13px;color:#7ee787">
            ✅ <b>{pdf_name}</b> indexed — {chunks} chunks ready · Ask your question below
        </div>
        """, unsafe_allow_html=True)

    # ── Render chat history ──
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # ── Chat input ──
    if prompt := st.chat_input(
        "Ask a question about your PDF..." if st.session_state.get("pdf_ready")
        else "Upload a PDF first to start asking questions..."
    ):
        if not st.session_state.get("pdf_ready"):
            st.warning("⚠️ Please upload a PDF first using the sidebar.")
            return

        user = st.session_state["user"]

        # Save & show user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        save_message(user, "user", prompt)
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Generate & show answer
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                answer, chunks = get_answer(prompt, st.session_state["retriever"])

            if not answer or answer.strip().lower() in ["unanswerable", "unknown", ""]:
                answer = "I couldn't find a clear answer in the PDF. Try rephrasing your question."

            st.markdown(answer)

            # Show source chunks
            with st.expander(f"📚 View {len(chunks)} source chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f'<div class="chunk-card"><div class="chunk-label">Chunk {i+1}</div>{chunk}</div>', unsafe_allow_html=True)

        # Save assistant message
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        save_message(user, "assistant", answer)

# ════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════
init_db()

st.set_page_config(
    page_title="DocChat — RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()

# Auth gate
if "user" not in st.session_state:
    show_auth()
    st.stop()

# Load chat history once per session
if "messages" not in st.session_state:
    st.session_state["messages"] = load_messages(st.session_state["user"])

# Show sidebar + main chat
show_sidebar()
show_chat()
