import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from ibm_watsonx_ai.foundation_models import ModelInference
import os
from dotenv import load_dotenv
import traceback
from collections import Counter
import spacy
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
import streamlit as st
import subprocess
import sys

load_dotenv()

def extract_glossary_terms(text, top_n=10):
    nlp = get_spacy_model()
    if nlp is None:
        raise RuntimeError("spaCy model not loaded. Cannot process text.")
    doc = nlp(text)
    # ... rest of your code ...


@st.cache_resource
def get_embedding_model():
    print("--> Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("--> Embedding model initialized.")
    return model

@st.cache_resource
def get_llm_model():
    print("--> Loading Watsonx credentials...")
    try:
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
        llm = ModelInference(
            model_id=model_id,
            params=generation_params,
            credentials=watsonx_credentials,
            project_id=project_id
        )
        print("--> SUCCESS: Watsonx LLM initialized!")
        return llm
    except Exception as e:
        print("\n\n" + "=" * 50)
        print("      AN ERROR OCCURRED DURING LLM INITIALIZATION")
        print(f"Error: {e}")
        print("=" * 50)
        traceback.print_exc()
        print("=" * 50 + "\n\n")
        return None

@st.cache_resource
def get_spacy_model():
    print("--> Initializing spaCy model...")
    try:
        nlp = spacy.load("./models/en_core_web_sm")
        print("--> spaCy model initialized from local models folder.")
        return nlp
    except OSError:
        print("Could not load local 'en_core_web_sm'. Trying global model...")
        try:
            nlp = spacy.load("en_core_web_sm")
            print("--> spaCy model initialized from global installation.")
            return nlp
        except OSError:
            print("Global spaCy model not found either. Please install it.")
            return None

def initialize_models():
    get_embedding_model()
    get_llm_model()
    get_spacy_model()

def extract_text_from_pdfs(pdf_files):
    full_text = ""
    for pdf_file in pdf_files:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    return full_text

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_embeddings_and_index(chunks):
    embedding_model = get_embedding_model()
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
    embedding_model = get_embedding_model()
    if not query or index is None or not embedding_model:
        return []
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
        valid_indices = [i for i in indices[0] if i < len(chunks)]
        return [chunks[i] for i in valid_indices]
    except Exception as e:
        print(f"Error during chunk retrieval: {e}")
        return []

def construct_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
    Answer the following question based only on the provided context. If the context does not contain the answer, state that the information is not available in the provided documents. Use External Knowledge if related info found else do not use any external knowledge.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    return prompt

def get_llm_answer(query, context_chunks):
    llm_model = get_llm_model()
    if not llm_model:
        return "Error: LLM model could not be initialized."
    prompt = construct_prompt(query, context_chunks)
    try:
        response = llm_model.generate_text(prompt=prompt)
        return response.strip()
    except Exception:
        traceback.print_exc()
        return "An error occurred while communicating with the LLM."

def summarize_text(text, max_chars=2000):
    llm_model = get_llm_model()
    if not llm_model:
        return "Error: LLM model not initialized."
    if not text.strip():
        return "No text to summarize."

    text = text[:max_chars]
    prompt = f"""
    Summarize the following text in a concise and clear manner:

    {text}

    Summary:
    """
    try:
        summary = llm_model.generate_text(prompt=prompt)
        return summary.strip()
    except Exception:
        traceback.print_exc()
        return "An error occurred while generating the summary."

def extract_glossary_terms(text, top_n=10):
    nlp = get_spacy_model()
    doc = nlp(text)
    terms = []

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        if chunk.root.pos_ in ["NOUN", "PROPN"] and len(chunk_text) > 2:
            if chunk_text not in STOP_WORDS and not all(char in punctuation for char in chunk_text):
                terms.append(chunk_text)

    freq = Counter(terms)
    return [term for term, _ in freq.most_common(top_n)]

def get_pdf_statistics(text):
    words = [word.lower() for word in text.split()]
    filtered_words = [
        word.strip(".,;:!?()[]{}\"'") for word in words
        if word.lower() not in STOP_WORDS and word.isalpha()
    ]

    total_words = len(filtered_words)
    unique_words = len(set(filtered_words))
    top_words = Counter(filtered_words).most_common(10)

    return {
        "Word Count": total_words,
        "Unique Words": unique_words,
        "Top 10 Words": top_words
    }
