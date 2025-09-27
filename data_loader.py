# data_loader.py
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_documents(data_path: str):
    """Loads documents from the specified path and splits them into chunks."""
    print(f"Loading documents from {data_path}...")
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Loaded and split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks