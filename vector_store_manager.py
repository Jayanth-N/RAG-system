# vector_store_manager.py
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings

def get_vector_store(store_path: str, embedding_model: str):
    """Loads an existing vector store or creates a new one if it doesn't exist."""
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=settings.GEMINI_API_KEY)
    
    if os.path.exists(store_path):
        print(f"Loading existing vector store from {store_path}...")
        return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # This part should be handled by the ingest script. 
        # We return None to indicate it needs to be created.
        print("Vector store not found.")
        return None

def create_and_save_vector_store(chunks, store_path: str, embedding_model: str):
    """Creates a new vector store from document chunks and saves it."""
    print(f"Creating new vector store at {store_path}...")
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=settings.GEMINI_API_KEY)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(store_path)
    print("Vector store created and saved successfully.")
    return vector_store