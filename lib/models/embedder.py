# lib/models/embedder.py

import os
from typing import List

from huggingface_hub import login
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class Embedder:
    """Простая обёртка над HuggingFaceEmbeddings."""

    def __init__(self, model_name: str | None = None) -> None:
        # 1. Логинимся в HuggingFace (если есть токен)
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token:
            login(token=token)

        # 2. Определяем имя модели
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "deepvk/USER-bge-m3",  # ваш дефолт
        )

        # 3. Создаём объект эмбеддингов
        self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Доступ к "сырому" объекту LangChain-эмбеддингов при необходимости."""
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг одного запроса."""
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Эмбеддинги списка документов."""
        return self._embeddings.embed_documents(texts)
