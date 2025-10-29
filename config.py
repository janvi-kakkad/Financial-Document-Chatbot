"""
Central configuration module for Financial RAG Chatbot.
All configurable parameters are centralized here for easy customization.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# DIRECTORY PATHS
# ============================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INDEX_FILE = os.path.join(DATA_DIR, "docs.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "ProsusAI/finbert")
EMBEDDING_DIMENSION = 768  # FinBERT outputs 768-dimensional vectors

# ============================================================================
# VECTOR DATABASE CONFIGURATION (Pinecone)
# ============================================================================
VECTOR_DB_PROVIDER = "pinecone"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "financial-rag")
PINECONE_CLOUD = "aws"
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_METRIC = "cosine"

# ============================================================================
# LLM CONFIGURATION (Google Gemini)
# ============================================================================
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL_PRIORITY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-pro"
]

# ============================================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
MIN_PDF_TEXT_LENGTH = 50

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================
DEFAULT_TOP_K = 5
MAX_CONTEXT_LENGTH = 6000

# ============================================================================
# FILE UPLOAD CONFIGURATION
# ============================================================================
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".txt"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB default
MAX_FILES_PER_REQUEST = 10

# ============================================================================
# API CONFIGURATION
# ============================================================================
MAX_QUERY_LENGTH = 1000
PORT = int(os.getenv("PORT", 5000))
DEBUG_MODE = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True")

# ============================================================================
# VALIDATION HELPERS
# ============================================================================
def validate_config():
    """
    Validate critical configuration settings and return warnings.
    Returns: List of warning messages
    """
    warnings = []
    
    if not PINECONE_API_KEY:
        warnings.append("PINECONE_API_KEY not set - vector search will be unavailable")
    
    if not GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY not set - LLM answer generation will be unavailable")
    
    if CHUNK_SIZE < 100:
        warnings.append(f"CHUNK_SIZE ({CHUNK_SIZE}) is very small - may affect quality")
    
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        warnings.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) >= CHUNK_SIZE ({CHUNK_SIZE}) - invalid configuration")
    
    return warnings

def get_config_summary():
    """Return a dictionary summarizing current configuration."""
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "vector_db": VECTOR_DB_PROVIDER,
        "pinecone_index": PINECONE_INDEX_NAME,
        "pinecone_region": PINECONE_REGION,
        "llm_provider": LLM_PROVIDER,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
    }
