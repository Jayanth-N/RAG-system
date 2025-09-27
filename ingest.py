# ingest.py
from config import settings
from data_loader import load_and_chunk_documents
from vector_store_manager import create_and_save_vector_store

def main():
    chunks = load_and_chunk_documents(settings.DATA_PATH)
    create_and_save_vector_store(chunks, settings.VECTOR_STORE_PATH, settings.EMBEDDING_MODEL)

if __name__ == "__main__":
    main()