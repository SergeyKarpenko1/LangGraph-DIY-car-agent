import os
import json
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from lib.models.embedder import Embedder
from lib.models.reranker import Reranker
from lib.utils.constant import USE_RERANKER, RERANK_TOP_K
from pydantic import ConfigDict

load_dotenv()


class OptionalRerankRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base: BaseRetriever
    use_reranker: bool = False
    top_k: int = 6
    reranker: Optional[Reranker] = None

    def __init__(self, base: BaseRetriever, use_reranker: bool = False, top_k: int = 6):
        reranker = Reranker() if use_reranker else None
        super().__init__(
            base=base, use_reranker=use_reranker, top_k=top_k, reranker=reranker
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base.invoke(query)
        if not self.use_reranker or not docs or self.reranker is None:
            return docs
        return self.reranker.rerank(query, docs, top_k=min(self.top_k, len(docs)))


def _docs_from_chroma(db: Chroma) -> List[Document]:
    try:
        raw = db._collection.get(include=["documents", "metadatas"])  # приватное API
    except Exception:
        return []

    docs: List[Document] = []
    for txt, md in zip(raw.get("documents", []) or [], raw.get("metadatas", []) or []):
        docs.append(Document(page_content=txt or "", metadata=md or {}))
    return docs


# --- embeddings ---
embeddings = Embedder().embeddings

# --- Chroma ---
persist_dir = os.getenv("CHROMA_PERSIST_DIR")
if not persist_dir:
    repo_root = Path(__file__).resolve().parents[2]
    candidate_dirs = [
        repo_root / "Data" / "processed" / "chromadb",
        repo_root / "Data" / "ChromaDB",
    ]

    def _looks_like_chroma(p: Path) -> bool:
        return (p / "chroma.sqlite3").exists()

    chosen = next((p for p in candidate_dirs if _looks_like_chroma(p)), None)
    if chosen is None:
        chosen = next(
            (p for p in candidate_dirs if p.exists() and any(p.iterdir())),
            candidate_dirs[-1],
        )
    persist_dir = str(chosen)

collection_name = os.getenv("CHROMA_COLLECTION_NAME", "VectorDB_deepvk_USER-bge-m3")

vectordb = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_dir,
)

# --- MMR ---
mmr = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 40, "lambda_mult": 0.5},
)

# --- BM25 (optional) ---
docs_for_bm25 = _docs_from_chroma(vectordb)
if docs_for_bm25:
    bm25 = BM25Retriever.from_documents(docs_for_bm25)
    bm25.k = 3
    base_retriever: BaseRetriever = EnsembleRetriever(
        retrievers=[mmr, bm25],
        weights=[0.6, 0.4],
    )
else:
    print(
        f"[retrieval_tool] BM25 disabled: no documents found in collection={collection_name!r} "
        f"persist_dir={persist_dir!r}"
    )
    base_retriever = mmr

retriever = OptionalRerankRetriever(
    base_retriever, use_reranker=USE_RERANKER, top_k=RERANK_TOP_K
)

retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_in_vectordb",
    description="Search and return information about car care, detailing and everything related to self-washing, cleaning.",
    response_format="content_and_artifact",
)
