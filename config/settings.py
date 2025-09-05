"""
Configuration settings for the Document Processing and RAG System.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    
    # Document processing settings
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_FORMATS: list = [".pdf", ".docx", ".doc", ".xlsx", ".pptx", ".txt", ".md", ".html", ".xhtml", ".csv", ".png", ".jpeg", ".jpg", ".tiff", ".bmp", ".webp", ".adoc", ".xml"]
    CHUNK_SIZE: int = 400  # Reduced for better sentence-based chunking
    CHUNK_OVERLAP: int = 100  # Increased overlap for better context preservation
    
    # LLM settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    
    # LEANN vector database settings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LEANN_INDEX_NAME: str = "main_collection"
    LEANN_BACKEND: str = "hnsw"  # Options: "hnsw", "diskann"
    LEANN_GRAPH_DEGREE: int = 32
    LEANN_COMPLEXITY: int = 64
    LEANN_USE_COMPACT: bool = True
    LEANN_USE_RECOMPUTE: bool = True
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8199
    API_DEBUG: bool = True
    
    # Timeout settings (in seconds) - Increased for new hybrid chunking strategy
    REQUEST_TIMEOUT: int = 600      # 10 minutes for general requests
    UPLOAD_TIMEOUT: int = 900       # 15 minutes for uploads (includes processing)
    PROCESSING_TIMEOUT: int = 900   # 15 minutes for document processing
    QUERY_TIMEOUT: int = 300        # 5 minutes for queries
    SYSTEM_OPERATION_TIMEOUT: int = 120  # 2 minutes for system operations
    
    # UI settings
    UI_HOST: str = "127.0.0.1"  # Use localhost instead of 0.0.0.0 to avoid accessibility issues
    UI_PORT: int = 7860
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Weights & Biases
    WANDB_PROJECT: str = "myr-ag-document-processing"
    WANDB_ENABLED: bool = False
    
    # MPS support for Apple Silicon
    USE_MPS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.DATA_DIR,
        settings.UPLOADS_DIR,
        settings.PROCESSED_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize directories
ensure_directories()
