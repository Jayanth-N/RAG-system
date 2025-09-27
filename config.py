# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # File Paths
    DATA_PATH: str = "data"
    VECTOR_STORE_PATH: str = "vector_db/faiss_index"

    # Model Configuration
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.0-flash"  # or "gemini-2.0-flash"
    
    # Retrieval Configuration
    RETRIEVER_K: int = 4 # Number of relevant chunks to retrieve

    GEMINI_API_KEY: str = "AIzaSyCoYKGpAeIl4BU0KeMrr4NT9OXRekzXKaQ"
    class Config:
        env_file = ".env"

settings = Settings()