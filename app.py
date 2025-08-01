import streamlit as st
from utils import (
    extract_text_from_pdfs,
    chunk_text,
    create_embeddings_and_index,
    retrieve_relevant_chunks,
    get_llm_answer
)

# --------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------
# Ensures that variables persist across user interactions (reruns).
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None


# --------------------------------------------------------------------------
# 3. SIDEBAR FOR DOCUMENT UPLOAD
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload Your Documents")
    st.markdown("""
    Upload your academic papers, textbooks, or notes in PDF format. 
    After uploading, click the **Process Documents** button.
    """)
    
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a few moments."):
                # Core processing pipeline
                raw_text = extract_text_from_pdfs(uploaded_files)
                text_chunks = chunk_text(raw_text)
                faiss_index, _ = create_embeddings_and_index(text_chunks)
                
                # Store processed data in session state
                if faiss_index is not None:
                    st.session_state.processed_data = {
                        "text_chunks": text_chunks,
                        "faiss_index": faiss_index
                    }
                    st.success("Documents processed successfully!")
                    st.info("You can now ask questions in the main panel.")
                else:
                    st.error("Failed to process documents. Please check your API credentials in the .env file and try again.")
        else:
            st.warning("Please upload at least one PDF file before processing.")


# --------------------------------------------------------------------------
# 4. MAIN APPLICATION LAYOUT
# --------------------------------------------------------------------------
st.title("📚 StudyMate: AI-Powered PDF Q&A System")
st.markdown("---")

# Display the question-answering interface only if documents are processed
if st.session_state.processed_data:
    st.header("Ask a Question")
    query = st.text_input(
        "Enter your question based on the uploaded content:", 
        key="query_input",
        placeholder="e.g., What is overfitting in machine learning?"
    )

    if st.button("Get Answer", type="primary"):
        if query:
            with st.spinner("Searching for the answer..."):
                # Retrieve processed data
                data = st.session_state.processed_data
                
                # Retrieve relevant context
                relevant_chunks = retrieve_relevant_chunks(
                    query, data["faiss_index"], data["text_chunks"]
                )
                
                if not relevant_chunks:
                    st.warning("Could not find relevant information in the documents to answer this question.")
                else:
                    # Generate the answer using the LLM
                    answer = get_llm_answer(query, relevant_chunks)
                    
                    # Store the Q&A pair and its context in history
                    st.session_state.qa_history.insert(0, {
                        "question": query,
                        "answer": answer,
                        "context": relevant_chunks
                    })
                    
                    # Rerun to clear the input box and display the new answer
                    st.rerun()
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload and process your documents using the sidebar to begin.")


# --------------------------------------------------------------------------
# 5. DISPLAY Q&A HISTORY
# --------------------------------------------------------------------------
if st.session_state.qa_history:
    st.markdown("---")
    st.header("Conversation History")
    
    for i, item in enumerate(st.session_state.qa_history):
        with st.container():
            st.markdown(f"**Q: {item['question']}**")
            st.markdown(f"**A:** {item['answer']}")
            
            with st.expander("Show Referenced Paragraphs"):
                for j, context_chunk in enumerate(item['context']):
                    st.markdown(f"**Reference {j+1}:**\n> {context_chunk}")
            
            st.markdown("---")