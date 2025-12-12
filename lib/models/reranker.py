# lib/models/reranker.py

import os
from typing import List, Tuple

import torch
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer

from langchain_core.documents import Document


class Reranker:
    """Простая обёртка над xProvence reranker."""

    def __init__(self, model_name: str | None = None) -> None:
        # 1. Логинимся в HuggingFace (если есть токен)
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token:
            login(token=token)

        # 2. Определяем имя модели
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL_NAME",
            "naver/xprovence-reranker-bgem3-v1",  # ваш дефолт
        )

        # 3. Создаём tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()

        # 4. Куда грузить модель (cpu/cuda/mps)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @property
    def raw_model(self) -> AutoModel:
        """Доступ к 'сырой' transformers-модели при необходимости."""
        return self.model

    @torch.no_grad()
    def score(self, query: str, passages: List[str]) -> List[float]:
        """Считает score для (query, passage) для каждого passage."""
        if not passages:
            return []

        pairs = [(query, p) for p in passages]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**inputs)

        # Обычно у reranker'ов есть logits; если вдруг нет — будет понятная ошибка.
        scores = out.logits.squeeze(-1).detach().float().cpu().tolist()
        return scores

    def rerank(self, query: str, docs: List[Document], top_k: int = 8) -> List[Document]:
        """Сортирует docs по релевантности к query и возвращает top_k."""
        if not docs:
            return []

        passages = [d.page_content for d in docs]
        scores = self.score(query, passages)

        scored: List[Tuple[float, Document]] = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [d for _, d in scored[:top_k]]