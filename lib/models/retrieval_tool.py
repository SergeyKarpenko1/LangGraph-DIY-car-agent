import os
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
        super().__init__(base=base, use_reranker=use_reranker, top_k=top_k, reranker=reranker)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base.invoke(query)
        if not self.use_reranker or not docs or self.reranker is None:
            return docs
        return self.reranker.rerank(query, docs, top_k=min(self.top_k, len(docs)))


def _docs_from_chroma(db: Chroma) -> List[Document]:
    raw = db._collection.get(include=["documents", "metadatas"])  # приватное API
    docs: List[Document] = []
    for txt, md in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=txt or "", metadata=md or {}))
    return docs


# --- embeddings ---
embeddings = Embedder().embeddings

# --- Chroma ---
persist_dir = "/Users/sergey/Desktop/Deteiling_agent/Data/ChromaDB"
collection_name = "VectorDB_deepvk_USER-bge-m3"

vectordb = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_dir,
)

# --- MMR ---
mmr = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5},
)

# --- BM25 ---
bm25 = BM25Retriever.from_documents(_docs_from_chroma(vectordb))

# --- Ensemble ---
ensemble = EnsembleRetriever(
    retrievers=[mmr, bm25],
    weights=[0.6, 0.4],
)

retriever = OptionalRerankRetriever(ensemble, use_reranker=USE_RERANKER, top_k=RERANK_TOP_K)

retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_in_vectordb",
    description="Search and return information about car care, detailing and everything related to self-washing, cleaning.",
    response_format="content_and_artifact",
)