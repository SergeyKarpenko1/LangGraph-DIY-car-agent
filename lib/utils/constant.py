import os

USE_RERANKER = os.getenv("USE_RERANKER", "1").lower() in ("1", "true", "yes")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))