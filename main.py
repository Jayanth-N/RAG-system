# main.py
from fastapi import FastAPI
from pydantic import BaseModel

from config import settings
from rag_chain import create_rag_chain
from vector_store_manager import get_vector_store
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the vector store and create the RAG chain on startup
vector_store = get_vector_store(settings.VECTOR_STORE_PATH, settings.EMBEDDING_MODEL)
if vector_store is None:
    raise ValueError("Vector store not found. Please run ingest.py first.")
    
rag_chain = create_rag_chain(vector_store, settings.LLM_MODEL, settings.RETRIEVER_K)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="An API for querying documents using Retrieval-Augmented Generation.",
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):
    response = rag_chain.invoke({"input": request.query})
    
    # Format the sources
    sources = []
    for doc in response["context"]:
        sources.append({
            "source": doc.metadata.get('source', 'N/A'),
            "page": doc.metadata.get('page', 'N/A')
        })

    return {
        "answer": response["answer"],
        "sources": sources
    }

# Optional: For running directly without an API for quick tests
if __name__ == "__main__":
    print("RAG system loaded. Ask a question.")
    while True:
        user_query = input("Your question: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = rag_chain.invoke({"input": user_query})
        print("\nAnswer:")
        print(response["answer"])
        print("\nSources:")
        for doc in response["context"]:
             print(f"  - Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
        print("\n" + "-"*50 + "\n")