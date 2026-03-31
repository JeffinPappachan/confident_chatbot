"""Application configuration for the Confident Group RAG chatbot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "confident_docs"
VECTOR_DB_PATH = BASE_DIR / "data" / "faiss_index"
LOG_PATH = Path(os.getenv("LOG_PATH", str(BASE_DIR / "logs" / "query_logs.json")))

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
HYBRID_RETRIEVAL_TOP_K = int(os.getenv("HYBRID_RETRIEVAL_TOP_K", "20"))
HYBRID_VECTOR_TOP_K = int(os.getenv("HYBRID_VECTOR_TOP_K", "15"))
HYBRID_BM25_TOP_K = int(os.getenv("HYBRID_BM25_TOP_K", "15"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
).strip()
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2200"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
QUERY_REWRITE_TIMEOUT_SECONDS = int(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "20"))
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
