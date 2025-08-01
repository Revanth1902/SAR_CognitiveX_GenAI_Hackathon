import streamlit as st
from utils import (
    extract_text_from_pdfs,
    chunk_text,
    create_embeddings_and_index,
    retrieve_relevant_chunks,
    get_llm_answer
)

# Inject dark theme and button styles via CSS
def local_css():
    st.markdown(
        """
        <style>
        /* Background and text */
        .main {
            background-color: #121212;
            color: #e0e0e0;
        }
        .css-1d391kg {
            background-color: #121212 !important;
            color: #e0e0e0 !important;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #14213d;
            color: #e0e0e0;
        }
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #bb86fc !important;
        }
        p{
        color: red !important;
        }
        /* Buttons */
        div.stButton > button {
            background-color: #26b170!im;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
            font-weight: 600;
            transition: background-color 0.3s ease;
            box-shadow: 0 2px 5px rgba(55,0,179,0.4);
        }
        div.stButton > button:hover {
            background-color: #6200ee;
            color: white;
        }
        /* Inputs */
        textarea, input[type="text"] {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #bb86fc;
            border-radius: 6px;
            padding: 8px;
        }
        /* Expanders */
        div[role="button"] {
            background-color: #2a2a2a !important;
            color: #bb86fc !important;
            border-radius: 6px;
            padding: 4px 12px;
        }
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #121212;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #bb86fc;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

local_css()

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide"
)

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

with st.sidebar:
    st.header("📂 Upload PDFs")
    st.write("Upload your academic papers or notes as PDFs. Then click **Process Documents**.")

    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once."
    )
    
    with st.form(key="process_form", clear_on_submit=False):
        process_btn = st.form_submit_button("Process Documents")
    
    if process_btn:
        if uploaded_files:
            with st.spinner("Processing documents, please wait..."):
                raw_text = extract_text_from_pdfs(uploaded_files)
                text_chunks = chunk_text(raw_text)
                faiss_index, _ = create_embeddings_and_index(text_chunks)

                if faiss_index:
                    st.session_state.processed_data = {
                        "text_chunks": text_chunks,
                        "faiss_index": faiss_index
                    }
                    st.success("✅ Documents processed successfully!")
                    st.info("Ask questions in the main panel now.")
                else:
                    st.error("❌ Failed to process documents. Check your API keys and try again.")
        else:
            st.warning("⚠️ Please upload at least one PDF before processing.")

st.title("📚 StudyMate: AI-Powered PDF Q&A")
st.write("---")

if st.session_state.processed_data:
    st.header("💬 Ask a Question")
    
    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_area(
            "Enter your question about the uploaded documents:",
            placeholder="E.g., What is overfitting in machine learning?",
            height=80
        )
        get_answer_btn = st.form_submit_button("Get Answer")

    if get_answer_btn:
        if query.strip():
            with st.spinner("Searching for answers..."):
                data = st.session_state.processed_data
                relevant_chunks = retrieve_relevant_chunks(query, data["faiss_index"], data["text_chunks"])
                
                if not relevant_chunks:
                    st.warning("No relevant info found in the documents to answer this.")
                else:
                    answer = get_llm_answer(query, relevant_chunks)
                    st.session_state.qa_history.insert(0, {
                        "question": query,
                        "answer": answer,
                        "context": relevant_chunks
                    })
                    st.experimental_rerun()
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload and process PDFs on the sidebar to get started.")

if st.session_state.qa_history:
    st.write("---")
    st.header("📝 Conversation History")
    for i, item in enumerate(st.session_state.qa_history):
        with st.expander(f"Q{i+1}: {item['question']}", expanded=False):
            st.write(f"**Answer:** {item['answer']}")
            with st.expander("Referenced Paragraphs"):
                for j, chunk in enumerate(item["context"], 1):
                    st.markdown(f"> {chunk}")
