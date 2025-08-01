import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# --- CHANGE 1: Import ModelInference instead of Model ---
from ibm_watsonx_ai.foundation_models import ModelInference
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

llm_model = None
embedding_model = None
try:
    print("--> Initializing embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("--> Embedding model initialized.")

    print("--> Loading Watsonx credentials...")
    watsonx_credentials = {
        "apikey": os.getenv("IBM_API_KEY"),
        "url": os.getenv("IBM_URL")
    }
    project_id = os.getenv("IBM_PROJECT_ID")
    print("--> Credentials loaded.")

    model_id = 'ibm/granite-13b-instruct-v2'

    generation_params = {
        "decoding_method": "greedy",
        "max_new_tokens": 300,
        "temperature": 0.5,
    }

    print("--> Initializing Watsonx LLM...")
    # --- CHANGE 2: Use ModelInference instead of Model ---
    llm_model = ModelInference(
        model_id=model_id,
        params=generation_params,
        credentials=watsonx_credentials,
        project_id=project_id
    )
    print("--> SUCCESS: Watsonx LLM initialized!")

except Exception:
    print("\n\n" + "="*50)
    print("      AN ERROR OCCURRED DURING INITIALIZATION")
    print("="*50)
    traceback.print_exc()
    print("="*50 + "\n\n")

# ... (the rest of the file remains the same) ...

def extract_text_from_pdfs(pdf_files):
    """Extracts text from a list of uploaded PDF files."""
    full_text = ""
    for pdf_file in pdf_files:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    return full_text

def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_embeddings_and_index(chunks):
    """Creates embeddings for text chunks and builds a FAISS index."""
    if not chunks or not embedding_model:
        return None, None
    try:
        embeddings = embedding_model.encode(chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        return index, np.array(embeddings, dtype=np.float32)
    except Exception as e:
        print(f"Error creating embeddings or FAISS index: {e}")
        return None, None

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    """Retrieves the most relevant chunks for a given query from the FAISS index."""
    if not query or index is None or not embedding_model:
        return []
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        valid_indices = [i for i in indices[0] if i < len(chunks)]
        retrieved_chunks = [chunks[i] for i in valid_indices]
        return retrieved_chunks
    except Exception as e:
        print(f"Error during chunk retrieval: {e}")
        return []

def construct_prompt(query, context_chunks):
    """Constructs the prompt for the LLM with context and instructions."""
    context = "\n\n".join(context_chunks)
    prompt = f"""
    Answer the following question based only on the provided context. If the context does not contain the answer, state that the information is not available in the provided documents. Do not use any external knowledge.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    return prompt

def get_llm_answer(query, context_chunks):
    """Gets a final answer from the Watsonx LLM."""
    if not llm_model:
        return "Error: LLM model could not be initialized. Please check the terminal logs for the detailed error message."
    
    prompt = construct_prompt(query, context_chunks)
    try:
        response = llm_model.generate_text(prompt=prompt)
        return response.strip()
    except Exception as e:
        print("Error during LLM call:")
        traceback.print_exc()
        return f"An error occurred while communicating with the LLM. Please check the terminal logs."