"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
# Provider dùng cho generation (pipeline), enrichment (M5) và RAGAS judge (M4).
# Ưu tiên Gemini nếu có GOOGLE_API_KEY, ngược lại dùng OpenAI. Xem src/llm.py.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# gemini-3.5-flash-lite: free tier có quota/ngày cao hơn hẳn gemini-3.6-flash (20 req/ngày)
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
# Judge của RAGAS dùng model KHÁC generator, vì:
#   1) quota free tier tính riêng từng model (500 req/ngày/model) → chia tải
#   2) tránh self-preference bias: LLM có xu hướng cho điểm cao cho output của chính nó
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
# Số worker RAGAS gọi LLM song song — giữ thấp để tránh 429 trên free tier.
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "2"))

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
