import os

# retrieval
USE_RERANKER = os.getenv("USE_RERANKER", "0").lower() in ("1", "true", "yes")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

# multystep_reasoning_agent
DOCS_KEY = "documents"
GO_FLAG = "proceed_to_generate"
MAX_REPHRASES = 1
