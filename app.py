import streamlit as st
import os

from utils import (
    extract_text_from_pdfs,
    chunk_text,
    create_embeddings_and_index,
    retrieve_relevant_chunks,
    get_llm_answer,
    summarize_text,
    extract_glossary_terms,
    get_pdf_statistics,
    initialize_models # Import the new initialization function
)

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

# Initialize models and cache them
initialize_models()

def enhanced_css():
    st.markdown("""
    <style>
    html, body, .main {
        background-color: #d6d3d4 !important;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    p{
                color:#d6d3d4}
    h1, h2, h3, h4 {
        color: #26b170;
    }

    .block-container {
        padding: 2rem 2rem;
                
    }

    [data-testid="stSidebar"] {
        background-color: #003C5F;
        color: whitesmoke;
    }

    div.stButton  {
    background: linear-gradient(90deg, #26b170, #34d399);
    color: black;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    box-shadow: 0px 4px 15px rgba(38, 177, 112, 0.3);
    cursor: pointer;
    transition: all 0.3s ease-in-out;
    animation: pulseGlow 2s infinite;
    width: 100%;
}

div.stButton {
    transform: scale(1.05);
    background: linear-gradient(90deg, #34d399, #26b170);
    box-shadow: 0px 6px 20px rgba(52, 211, 153, 0.4);
}


    textarea, input[type="text"] {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #34d399;
        border-radius: 6px;
        padding: 10px;
        width: 100%;
        resize: vertical;
    }

    .divider {
        margin: 2rem 0;
        height: 1px;
        background: linear-gradient(to right, #34d399, #26b170);
        border: none;
    }

    
    .scrollable {
        overflow-y: auto;
        max-height: 250px;
        padding-right: 0.5rem;
    }


    </style>
    """, unsafe_allow_html=True)

enhanced_css()

# === Session States ===
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

# === Sidebar ===
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Select PDF files", type="pdf", accept_multiple_files=True
    )

    with st.form(key="process_form"):
        process_btn = st.form_submit_button("✨ Process Documents")

    if process_btn:
        if uploaded_files:
            with st.spinner("⏳ Analyzing and chunking your documents..."):
                raw_text = extract_text_from_pdfs(uploaded_files)
                text_chunks = chunk_text(raw_text)
                faiss_index, _ = create_embeddings_and_index(text_chunks)

                if faiss_index:
                    st.session_state.processed_data = {
                        "text_chunks": text_chunks,
                        "faiss_index": faiss_index,
                        "raw_text": raw_text
                    }
                    st.success("✅ Documents processed successfully!")
                else:
                    st.error("❌ Failed to process documents.")
        else:
            st.warning("⚠️ Please upload at least one PDF.")

st.title("📚 StudyMate: AI-Powered PDF Q&A")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Layout 2 columns per row for 4 sections
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("💬 Ask Questions")

    current_answer = None  # Track the most recent answer for display

    if st.session_state.processed_data:
        with st.form(key="query_form", clear_on_submit=True):
            query = st.text_area("🔎 What would you like to ask?", height=100)
            get_answer_btn = st.form_submit_button("💡 Get Answer")

        if get_answer_btn:
            if query.strip():
                with st.spinner("💭 Thinking..."):
                    data = st.session_state.processed_data
                    relevant_chunks = retrieve_relevant_chunks(query, data["faiss_index"], data["text_chunks"])
                    if relevant_chunks:
                        current_answer = get_llm_answer(query, relevant_chunks)
                        st.session_state.qa_history.insert(0, {
                            "question": query,
                            "answer": current_answer,
                            "context": relevant_chunks
                        })
                    else:
                        current_answer = "⚠️ No relevant information found."
            else:
                st.warning("⚠️ Please enter a valid question.")

        # Display answer immediately below the input
        if current_answer:
            st.markdown("#### ✅ Answer:")
            st.success(current_answer)

    else:
        st.info("📌 Upload and process PDFs to start asking questions.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📝 Document Summary")
    if st.session_state.processed_data:
        with st.spinner("📚 Summarizing..."):
            summary = summarize_text(st.session_state.processed_data["raw_text"])
            st.markdown(f'<div class="scrollable">{summary}</div>', unsafe_allow_html=True)
    else:
        st.info("📌 Upload and process PDFs to see the summary.")
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📘 Glossary Terms")
    if st.session_state.processed_data:
        with st.spinner("🔍 Extracting key terms..."):
            terms = extract_glossary_terms(st.session_state.processed_data["raw_text"])
            for term in terms:
                st.markdown(f"🔹 **{term}**")
    else:
        st.info("📌 Upload and process PDFs to generate glossary.")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📊 Document Insights")
    if st.session_state.processed_data:
        stats = get_pdf_statistics(st.session_state.processed_data["raw_text"])
        st.markdown(f"**📄 Word Count:** `{stats['Word Count']}`")
        st.markdown(f"**🔣 Unique Words:** `{stats['Unique Words']}`")
        st.markdown("**📈 Top 10 Frequent Words:**")
        for word, count in stats["Top 10 Words"]:
            st.markdown(f"• `{word}`: {count}")
    else:
        st.info("📌 Upload and process PDFs to see insights.")
    st.markdown('</div>', unsafe_allow_html=True)

# === Conversation History below ===
if st.session_state.qa_history:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("📜 Conversation History")
    for i, item in enumerate(st.session_state.qa_history):
        with st.expander(f"🗨️ Q{i+1}: {item['question']}"):
            st.markdown(f"**Answer:** {item['answer']}")
            with st.expander("📚 Referenced Content"):
                for j, chunk in enumerate(item["context"], 1):
                    st.markdown(f"> {chunk}")