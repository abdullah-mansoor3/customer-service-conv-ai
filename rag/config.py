from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DATA_DIR = PROJECT_ROOT / "rag_data"
CHROMA_DIR = PROJECT_ROOT / "rag" / "chroma_store"
COLLECTION_NAME = "isp_support_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_TOKEN_SIZE = 400
CHUNK_TOKEN_OVERLAP = 50

DEFAULT_RETRIEVE_TOP_K = 10
DEFAULT_RERANK_TOP_K = 3
DEFAULT_SENTENCES_PER_CHUNK = 3

RELEVANCE_THRESHOLD = 0.30
SEMANTIC_CACHE_THRESHOLD = 0.92
SEMANTIC_CACHE_CAPACITY = 50

DEFAULT_LLAMACPP_ENDPOINT = "http://localhost:8080/v1/chat/completions"
