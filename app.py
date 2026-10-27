import os

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from streamlit_firebase_auth import FirebaseAuth

from services.answer_generator import generate_answer
from services.document_pipeline import process_pdf
from services.firebase_auth import verify_firebase_token
from services.qdrant_store import delete_document, search_documents, store_document


# --------------------------------------------------
# Page setup and visual style
# --------------------------------------------------
st.set_page_config(
    page_title="NotebookLM Lite",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --page: #101218;
        --sidebar: #151820;
        --surface: #1a1e28;
        --surface-raised: #202633;
        --ink: #eef1f8;
        --muted: #a1a9ba;
        --line: #303747;
        --accent: #a8c7fa;
        --accent-strong: #7cacf8;
        --accent-soft: #253650;
        --green: #9ad9b4;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Google Sans', 'Inter', sans-serif;
        color: var(--ink);
    }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--page); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(16,18,24,.92); }
    [data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    .block-container { max-width: 1220px; padding-top: 1.8rem; padding-bottom: 3rem; }

    /* Keep all Streamlit text readable in dark mode. */
    p, label, li, .stMarkdown, [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stCaptionContainer"], .small-label { color: var(--muted) !important; }
    h1, h2, h3, h4 { color: #f3f5fb !important; letter-spacing: -.035em; }
    [data-testid="stWidgetLabel"] p { color: #d9deea !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #d8deeb; }

    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom:1.35rem; }
    .brand-icon { display:flex; align-items:center; justify-content:center; width:44px; height:44px;
        border-radius:15px; background:linear-gradient(145deg,#344c70,#25364e); border:1px solid #4b6282;
        font-size:23px; box-shadow:0 7px 22px rgba(0,0,0,.22); }
    .brand-name { font-size:1.15rem; font-weight:700; letter-spacing:-.45px; color:#f3f5fb; }
    .brand-caption { color:#9da9bd; font-size:.82rem; margin-top:3px; }

    .hero { padding:1.25rem 0 1.35rem; }
    .eyebrow { color:var(--accent); font-size:.75rem; font-weight:700; letter-spacing:1.8px; text-transform:uppercase; }
    .hero h1 { font-size:clamp(2.1rem,4vw,3.1rem); line-height:1.13; font-weight:500; letter-spacing:-1.5px; margin:.6rem 0 .8rem; color:#f3f5fb; }
    .hero p { color:var(--muted); font-size:1rem; max-width:720px; line-height:1.75; }
    .soft-card { background:linear-gradient(145deg,#1c2230,#181c25); border:1px solid #343c4d; border-radius:22px; padding:1.3rem 1.35rem; box-shadow:0 12px 32px rgba(0,0,0,.16); }
    .section-title { font-size:1.02rem; font-weight:650; color:#f0f3fa; margin-bottom:.25rem; }
    .section-subtitle { font-size:.87rem; color:var(--muted); margin-bottom:.9rem; line-height:1.5; }
    .doc-pill { display:inline-block; padding:5px 10px; background:var(--accent-soft); color:var(--accent); border:1px solid #354e70; border-radius:999px; font-size:.78rem; margin:3px 4px 3px 0; }

    /* Cards, chat and source panels */
    [data-testid="stChatMessage"] { background:#1b202b; border:1px solid #303849; border-radius:18px; padding:1rem 1.15rem; margin:.65rem 0; }
    [data-testid="stChatMessage"] p { color:#e8ecf5; line-height:1.75; }
    [data-testid="stChatInput"] textarea, [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        background:#171b24 !important; color:#f0f3fa !important; border-color:#3a4355 !important; border-radius:13px !important;
    }
    textarea::placeholder, input::placeholder { color:#7f899d !important; }
    [data-testid="stChatInput"] { border-radius:17px; }
    [data-testid="stChatInput"] textarea:focus, [data-testid="stTextInput"] input:focus { border-color:var(--accent-strong) !important; box-shadow:0 0 0 1px var(--accent-strong) !important; }

    div[data-testid="stFileUploader"] { background:#191e29; border:1px dashed #56647b; border-radius:18px; padding:14px; }
    div[data-testid="stFileUploader"] section { background:transparent; border:0; }
    div[data-testid="stFileUploader"] button { background:#28364b !important; color:#e8efff !important; border:1px solid #465a78 !important; border-radius:11px !important; }
    div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span { color:#b6bfd0 !important; }

    .stButton > button, [data-testid="stFormSubmitButton"] button {
        border-radius:12px; min-height:2.7rem; font-weight:650; padding:.45rem 1rem;
        background:#272e3b; color:#e9edf6; border:1px solid #424c60;
        transition:all .16s ease; box-shadow:0 3px 10px rgba(0,0,0,.12);
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
        background:#333e50; border-color:#6a7c99; color:#fff; transform:translateY(-1px);
    }
    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background:linear-gradient(135deg,#b7d0fb,#8db8fa); color:#152238; border:1px solid #b7d0fb;
        box-shadow:0 5px 18px rgba(108,157,231,.18);
    }
    .stButton > button[kind="primary"]:hover { background:#c8dcff; color:#111d30; border-color:#c8dcff; }

    div[data-testid="stExpander"] { background:#191e28; border:1px solid #323b4c; border-radius:14px; overflow:hidden; }
    div[data-testid="stExpander"] summary { color:#e8edf7; }
    [data-testid="stMetric"] { background:#202633; border:1px solid #343d4e; padding:1rem; border-radius:15px; }
    [data-testid="stMetricLabel"] { color:#aab4c7 !important; }
    [data-testid="stMetricValue"] { color:#f2f5fc !important; }
    [data-testid="stAlert"] { background:#202735; border:1px solid #39465b; color:#e9edf6; border-radius:13px; }
    [data-testid="stNotification"] { border-radius:12px; }
    [data-testid="stStatusWidget"] { background:#202633; }
    button[kind="header"] { color:#dfe6f4 !important; }
    [data-testid="stToolbar"] { right:1rem; }
    hr { border-color:var(--line); }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()

# --------------------------------------------------
# Firebase authentication
# --------------------------------------------------
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
}

auth = FirebaseAuth(firebase_config)
session = auth.check_session()

if not session:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-icon">📚</div>
          <div><div class="brand-name">NotebookLM Lite</div>
          <div class="brand-caption">Your personal research workspace</div></div>
        </div>
        <div class="hero">
          <div class="eyebrow">Read less. Understand more.</div>
          <h1>Your documents, ready to talk.</h1>
          <p>Sign in to explore your PDFs, ask questions, and find answers grounded in your own sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.subheader("Welcome to your notebook")
        st.write("Sign in with Google to open your private research workspace.")
        auth.login_form(providers=["google"])
    st.stop()

try:
    token = session.get("firebaseIdToken")
    if not token:
        st.error("Your sign-in session has expired. Please sign in again.")
        auth.logout_form()
        st.stop()
    current_user = verify_firebase_token(token)
except Exception:
    st.error("Authentication failed. Please sign in again.")
    auth.logout_form()
    st.stop()

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.get("active_user_id") != current_user["uid"]:
    st.session_state.active_user_id = current_user["uid"]
    st.session_state.processed_documents = {}
    st.session_state.chat_history = []
    st.rerun()

# --------------------------------------------------
# Sidebar: notebook and document controls
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-icon">📚</div>
          <div><div class="brand-name">NotebookLM Lite</div>
          <div class="brand-caption">Personal research space</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(current_user.get("email", "Signed in"))
    auth.logout_form()
    st.divider()
    st.markdown("### Your sources")

    available_documents = list(st.session_state.processed_documents.keys())
    if available_documents:
        selected_documents = st.multiselect(
            "Choose sources for chat",
            options=available_documents,
            default=available_documents,
            help="Only selected PDFs will be searched.",
        )
        st.caption(f"{len(available_documents)} source(s) in this notebook")

        with st.expander("Manage sources"):
            document_to_delete = st.selectbox(
                "Choose a PDF to remove",
                options=available_documents,
                key="document_to_delete",
            )
            if st.button("Remove selected PDF", use_container_width=True):
                try:
                    delete_document(
                        user_id=current_user["uid"],
                        filename=document_to_delete,
                    )
                    del st.session_state.processed_documents[document_to_delete]
                    st.session_state.chat_history = []
                    st.success("PDF removed from your notebook.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not remove PDF: {error}")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        selected_documents = []
        st.info("Add a PDF to start your notebook.")

    st.divider()
    st.caption("Private workspace · Powered by Gemini and Qdrant")

# --------------------------------------------------
# Main workspace
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Your AI research workspace</div>
      <h1>What would you like to learn?</h1>
      <p>Add your PDFs, then ask questions. NotebookLM Lite searches your selected sources and gives you answers with references.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, info_col = st.columns([1.5, 1], gap="large")
with upload_col:
    st.markdown('<div class="section-title">Add sources</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Upload one or more PDF documents to this notebook.</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here or browse files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} PDF(s) selected")
        with st.expander("Preview selected PDFs"):
            for uploaded_file in uploaded_files:
                try:
                    with pymupdf.open(stream=uploaded_file.getvalue(), filetype="pdf") as pdf:
                        st.markdown(f"**{uploaded_file.name}** · {len(pdf)} page(s)")
                except Exception as error:
                    st.error(f"Could not preview {uploaded_file.name}: {error}")
        if st.button("＋ Process PDFs", type="primary", use_container_width=True):
            successful_documents = 0
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                if filename in st.session_state.processed_documents:
                    st.warning(f"'{filename}' is already in this notebook.")
                    continue
                try:
                    with st.spinner(f"Reading {filename}..."):
                        records = process_pdf(
                            pdf_bytes=uploaded_file.getvalue(),
                            filename=filename,
                        )
                        if not records:
                            st.warning(f"No readable content was found in {filename}.")
                            continue
                        store_document(user_id=current_user["uid"], records=records)
                        st.session_state.processed_documents[filename] = {
                            "chunks": len(records),
                            "size": len(uploaded_file.getvalue()),
                        }
                    successful_documents += 1
                    st.success(f"Added {filename} · {len(records)} text chunks")
                except Exception as error:
                    st.error(f"Could not process {filename}: {error}")
            if successful_documents:
                st.session_state.chat_history = []
                st.rerun()

with info_col:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Your notebook at a glance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Everything you need to start exploring.</div>', unsafe_allow_html=True)
    st.metric("Sources added", len(st.session_state.processed_documents))
    total_chunks = sum(item.get("chunks", 0) for item in st.session_state.processed_documents.values())
    st.metric("Searchable text chunks", total_chunks)
    st.markdown("<span class='small-label'>Tip: ask focused questions for more useful answers.</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown("### Conversation")

if not st.session_state.processed_documents:
    st.info("Your conversation will appear here after you add and process a PDF.")
elif not selected_documents:
    st.warning("Select at least one source in the sidebar to start chatting.")
else:
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            if item.get("sources"):
                with st.expander(f"View sources · {len(item['sources'])}"):
                    for index, source in enumerate(item["sources"], start=1):
                        st.markdown(f"**{index}. {source['filename']}** · Page {source['page_number']}")
                        st.write(source["text"])
                        st.divider()

    question = st.chat_input("Ask a question about your sources...")
    if question:
        try:
            with st.spinner("Searching your sources..."):
                results = search_documents(
                    user_id=current_user["uid"],
                    question=question,
                    limit=5,
                    filenames=selected_documents,
                )
                if results:
                    answer = generate_answer(question, results)
                else:
                    answer = "I couldn't find relevant information in the selected sources. Try rephrasing your question."
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": results,
            })
            st.rerun()
        except Exception as error:
            st.error(f"Could not generate an answer: {error}")

st.markdown("<br><div style='text-align:center;color:#80868b;font-size:.78rem'>NotebookLM Lite · Your sources, your research</div>", unsafe_allow_html=True)
