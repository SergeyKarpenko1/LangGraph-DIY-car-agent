import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)

from langchain_core.documents import Document
from langchain_core.tools import tool

from lib.models.embedder import Embedder

load_dotenv()

embedder = Embedder()  # можно передать model_name="..." при желании
embeddings = embedder.embeddings  # это HuggingFaceEmbeddings из LangChain

md_folder = "Data/raw/articles"

# Используем загрузчик Markdown
loader = DirectoryLoader(
    path=md_folder,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    loader_kwargs={"mode": "single"},  # или "elements", если не нужны элементы
)

# Загружаем документы
docs: list[Document] = loader.load()

# Разбиваем на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=150, separators=["\n\n", "\n", ".", " "]
)

# Разбиваем на чанки
recursive_chunks: list[Document] = text_splitter.split_documents(docs)

# print(recursive_chunks[0])

persist_dir = "Data/processed/chromadb"

vectordb = Chroma(
    collection_name="VectorDB_deepvk_USER-bge-m3",
    embedding_function=embeddings,  # тот же эмбеддер, что использовался при создании
    persist_directory=persist_dir,
)


def _stable_id(doc: Document, idx: int) -> str:
    """Устойчивый ID по источнику и номеру чанка.
    Хеш используем, чтобы избежать проблем со спецсимволами/длиной пути в raw-id.
    """
    src = (doc.metadata or {}).get("source") or (doc.metadata or {}).get("file") or ""
    src = str(Path(src)).replace("\\", "/")
    key = f"{src}::{idx:06d}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# Добавляем документы только один раз — если коллекция пуста
if vectordb._collection.count() == 0:  # приватное API, но надёжно для этой проверки
    docs_to_add, ids = [], []
    for i, d in enumerate(recursive_chunks):
        md = dict(d.metadata or {})
        # нормализуем и фиксируем полезные поля
        md.setdefault("source", md.get("file") or md.get("path") or "md")
        md["chunk"] = i

        docs_to_add.append(Document(page_content=d.page_content, metadata=md))
        ids.append(_stable_id(d, i))

    vectordb.add_documents(docs_to_add, ids=ids)
    # vectordb.persist()  # опционально: в новых версиях вызов не обязателен
