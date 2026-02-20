import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama or groq
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:70b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding Model
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
CHROMA_KNOWLEDGE_COLLECTION = "knowledge_base"
CHROMA_ASSESSMENT_COLLECTION = "assessment_bank"

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Letta
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY", "")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# PostgreSQL
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost/ai_tutor")

# KG settings
KG_VISIBLE_THRESHOLD = 1  # show KG when node count > this value

# RAG settings
RAG_TOP_K = 5
RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 50

# Assessment settings
MAX_ASSESSMENT_ATTEMPTS = 3
