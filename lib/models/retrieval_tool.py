from typing import List


from dotenv import load_dotenv

from langchain.retrievers import  EnsembleRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

from lib.models.embedder import Embedder

load_dotenv()

embedder = Embedder()  # можно передать model_name="..." при желании
embeddings = embedder.embeddings  # это HuggingFaceEmbeddings из LangChain

# Параметры данных/индекса
md_folder = "/Users/sergey/Desktop/Deteiling_agent/Data/cleaned"
persist_dir = "/Users/sergey/Desktop/Deteiling_agent/Data/ChromaDB"
collection_name = "VectorDB_deepvk_USER-bge-m3"

# Открываем (или создаём пустую) коллекцию Chroma
vectordb = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,      # важно: тот же эмбеддер, что использовался при создании
    persist_directory=persist_dir,
)


def _docs_from_chroma(db: Chroma) -> List[Document]:
    """Забираем все документы (чанки) из существующей коллекции Chroma."""
    raw = db._collection.get(include=["documents", "metadatas"])  # приватное API
    docs: List[Document] = []
    for txt, md in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=txt or "", metadata=md or {}))
    return docs


# --- 3) Ретривер MMR (Chroma) ---
mmr = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 40, "lambda_mult": 0.5},
)

# --- 4) Ретривер BM25 (из чанков, восстановленных из Chroma) ---
chroma_docs = _docs_from_chroma(vectordb)
bm25 = BM25Retriever.from_documents(chroma_docs)

# --- 5) Ensemble ---
ensemble = EnsembleRetriever(
    retrievers=[mmr, bm25],
    weights=[0.6, 0.4],
)


retriever_tool = create_retriever_tool(
    ensemble,
    name="retrieve_in_vectordb",
    description="Search and return information about car care, detailing and everything related to self-washing, cleaning.",
    response_format="content_and_artifact",
)
