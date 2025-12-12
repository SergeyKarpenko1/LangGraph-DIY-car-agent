import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document


class SimpleChromaBuilder:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def build_or_load(
        self,
        md_folder: str,
        persist_dir: str,
        collection_name: str,
        *,
        chunk_size: int = 1200,          # <-- добавили
        chunk_overlap: int = 150,        # (оставил тоже параметром — полезно, но можно убрать)
    ) -> Chroma:
        # 1) открыть/создать коллекцию
        db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        # 2) если не пустая — просто вернуть
        if db._collection.count() > 0:  # приватное API (как у вас)
            return db

        # 3) загрузить md
        loader = DirectoryLoader(
            path=md_folder,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            loader_kwargs={"mode": "single"},
        )
        docs: list[Document] = loader.load()

        # 4) порезать на чанки (теперь размер задаётся параметром)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks: list[Document] = splitter.split_documents(docs)

        # 5) добавить чанки
        docs_to_add, ids = [], []
        for i, d in enumerate(chunks):
            md = dict(d.metadata or {})
            src = md.get("source") or md.get("file") or md.get("path") or "md"
            src = str(Path(src)).replace("\\", "/")
            md["source"] = src
            md["chunk"] = i

            docs_to_add.append(Document(page_content=d.page_content, metadata=md))
            ids.append(hashlib.sha1(f"{src}::{i:06d}".encode("utf-8")).hexdigest())

        db.add_documents(docs_to_add, ids=ids)
        return db

# пример создания БД
#     
# builder = SimpleChromaBuilder(embeddings)

# vectordb = builder.build_or_load(
#     md_folder="Data/raw/articles",
#     persist_dir="Data/processed/chromadb",
#     collection_name="VectorDB_deepvk_USER-bge-m3",
#     chunk_size=800,        # <-- настраиваете здесь
#     chunk_overlap=120,
# )   