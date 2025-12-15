# lib/models/reranker.py

import os
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer
from langchain_core.documents import Document


class Reranker:
    """Минимальный reranker: считает скор и сортирует документы."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL_NAME",
            "naver/xprovence-reranker-bgem3-v1",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)

    @torch.no_grad()
    def _score(self, query: str, passages: List[str]) -> List[float]:
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

        # xProvence обычно отдаёт ranking_scores; оставим небольшую страховку
        if hasattr(out, "ranking_scores"):
            scores = out.ranking_scores
        elif hasattr(out, "reranking_scores"):
            scores = out.reranking_scores
        elif hasattr(out, "logits"):
            scores = out.logits.squeeze(-1)
        else:
            raise RuntimeError(
                "Model output does not contain ranking scores (ranking_scores/logits)."
            )

        scores = scores.detach().float().cpu().tolist()
        return [float(s) for s in scores]

    def rerank(
        self, query: str, docs: List[Document], top_k: int = 5
    ) -> List[Document]:
        if not docs:
            return []

        passages = [d.page_content or "" for d in docs]
        scores = self._score(query, passages)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]
