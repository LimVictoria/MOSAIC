import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

# ─── LLM ───
LLM_PROVIDER = get_secret("LLM_PROVIDER", "groq")
LLM_MODEL    = get_secret("LLM_MODEL", "llama3-70b-8192")
GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
OLLAMA_BASE_URL = get_secret("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Embedding ───
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ─── ChromaDB ───
CHROMA_PERSIST_DIR = get_secret("CHROMA_PERSIST_DIR", "./data/chromadb")
CHROMA_KNOWLEDGE_COLLECTION = "knowledge_base"
CHROMA_ASSESSMENT_COLLECTION = "assessment_bank"

# ─── Neo4j ───
NEO4J_URI      = get_secret("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = get_secret("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "password")

# ─── Letta ───
LETTA_BASE_URL = get_secret("LETTA_BASE_URL", "https://inference.letta.com")
LETTA_API_KEY  = get_secret("LETTA_API_KEY", "")

# ─── RAG ───
RAG_TOP_K         = 5
RAG_CHUNK_SIZE    = 512
RAG_CHUNK_OVERLAP = 50

# ─── KG ───
KG_VISIBLE_THRESHOLD = 1

# ─── Assessment ───
MAX_ASSESSMENT_ATTEMPTS = 3
