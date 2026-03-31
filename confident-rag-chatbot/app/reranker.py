"""Chunk reranking using a cross-encoder model."""

from __future__ import annotations

import logging
from threading import Lock
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import ENABLE_RERANKER, RERANKER_MODEL, RERANKER_TOP_K


logger = logging.getLogger(__name__)
_reranker_model: CrossEncoder | None = None
_reranker_lock = Lock()


def _get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                logger.info("Loading reranker model...")
                _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


class Reranker:
    """Rerank retrieved chunks by cross-encoder relevance scores."""

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        if not ENABLE_RERANKER:
            return documents[:RERANKER_TOP_K]

        try:
            model = _get_reranker_model()
            pairs = [[query, document.page_content] for document in documents]
            scores = model.predict(pairs)
            ranked = sorted(
                zip(documents, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            return [document for document, _ in ranked[:RERANKER_TOP_K]]
        except Exception as exc:
            logger.exception(
                "Reranker failed; falling back to original retrieval order: %s",
                exc,
            )
            return documents[:RERANKER_TOP_K]
