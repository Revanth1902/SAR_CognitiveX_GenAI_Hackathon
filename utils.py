import fitz #for pdf open nd read
from sentence_transformers import SentenceTransformer #convert text to vectors
import faiss #serach for vectors similarty
import numpy as np #work with arrays
from ibm_watsonx_ai.foundation_models import ModelInference #ibmmodel
import os #read environment variables, paths
from dotenv import load_dotenv #to get env
import traceback #debugging

load_dotenv()

llm_model = None
embedding_model = None
try:
    print("--> Initializing embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # turns text into vectors 
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


#take all pdfs and reach each and extrct all text into a large string and returns in it 
def extract_text_from_pdfs(pdf_files):
    full_text = ""
    for pdf_file in pdf_files:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    return full_text

# Returns a list of these chunks.

#long textt splited into small pices and overlap and return list
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


#the list (chunks) and use model to convert into vector and use faiss to create index and return index and array 
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

#serached quey go as input and using the faiss most relevant chunk is returned
def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    
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

#building a prompt with context(chunks) and query
def construct_prompt(query, context_chunks):
    """Constructs the prompt for the LLM with context and instructions."""
    context = "\n\n".join(context_chunks)
    prompt = f"""
    Answer the following question based only on the provided context. If the context does not contain the answer, state that the information is not available in the provided documents. Use External Knowledge if realted info found else do not use any external knowledge.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    return prompt


#sends the prompt to the Watsonx LLM to get a generated answer
def get_llm_answer(query, context_chunks):
   
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