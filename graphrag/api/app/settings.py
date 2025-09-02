import os

# Google Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # For processing (embeddings, chunking, etc.)

# Separate API key and model for final response generation
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY", GEMINI_API_KEY)  # Falls back to main key if not set
RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "gemini-2.0-flash-exp")  # Model for final responses

CHROMA_URL = os.getenv("CHROMA_URL", "http://localhost:8000")

# Neo4j GraphRAG settings
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# Processing models (used for embeddings, chunking, entity extraction, etc.)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-pro")  # For processing tasks
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/embedding-001")

# retrieval / context knobs
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
COLLECTION = os.getenv("COLLECTION", "thedrive")

# GraphRAG settings
ENABLE_GRAPHRAG = os.getenv("ENABLE_GRAPHRAG", "1") == "1"
