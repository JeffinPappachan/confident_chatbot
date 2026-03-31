"""Hybrid retrieval using FAISS vector search plus BM25 keyword search."""

from __future__ import annotations

from collections.abc import Iterable
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import HYBRID_BM25_TOP_K, HYBRID_RETRIEVAL_TOP_K, HYBRID_VECTOR_TOP_K
from app.document_loader import normalize_metadata
from app.vector_store import FAISSVectorStore


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _normalize_document(document: Document) -> Document:
    metadata = normalize_metadata(document.metadata or {})
    return Document(page_content=document.page_content, metadata=metadata)


def _build_chunk_id(document: Document) -> str:
    metadata = document.metadata or {}
    explicit_id = metadata.get("chunk_id")
    if explicit_id:
        return str(explicit_id)

    source = metadata.get("source", "unknown")
    page = metadata.get("page", "unknown")
    snippet = document.page_content.strip()[:120]
    return f"{source}|{page}|{snippet}"


class HybridRetriever:
    """Retrieve documents from FAISS and BM25, then merge results."""

    def __init__(self, vector_store: FAISSVectorStore) -> None:
        self.vector_store = vector_store
        self._documents_cache: list[Document] | None = None
        self._bm25: BM25Okapi | None = None

    def retrieve(self, query: str) -> List[Document]:
        rewritten_query = query.strip()
        if not rewritten_query:
            return []

        query_embedding = self.vector_store.embeddings.embed_query(rewritten_query)
        vector_documents = self.vector_store.similarity_search_by_vector(
            query_embedding,
            k=HYBRID_VECTOR_TOP_K,
        )
        bm25_documents = self._bm25_search(rewritten_query, k=HYBRID_BM25_TOP_K)
        return self._merge_results(vector_documents, bm25_documents, limit=HYBRID_RETRIEVAL_TOP_K)

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        documents = self._get_all_documents()
        if not documents:
            return []

        bm25 = self._get_bm25()
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        matches: list[Document] = []
        for index in ranked_indices:
            if scores[index] <= 0:
                continue
            matches.append(documents[index])
            if len(matches) >= k:
                break

        return matches

    def _merge_results(
        self,
        primary: Iterable[Document],
        secondary: Iterable[Document],
        *,
        limit: int,
    ) -> List[Document]:
        merged: list[Document] = []
        seen_ids: set[str] = set()

        for raw_document in list(primary) + list(secondary):
            document = _normalize_document(raw_document)
            chunk_id = _build_chunk_id(document)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            metadata = dict(document.metadata or {})
            metadata.setdefault("chunk_id", chunk_id)
            merged.append(Document(page_content=document.page_content, metadata=metadata))
            if len(merged) >= limit:
                break

        return merged

    def _get_all_documents(self) -> list[Document]:
        if self._documents_cache is None:
            self._documents_cache = [
                _normalize_document(document)
                for document in self.vector_store.get_all_documents()
            ]
        return self._documents_cache

    def _get_bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            documents = self._get_all_documents()
            tokenized_corpus = [_tokenize(document.page_content) for document in documents]
            self._bm25 = BM25Okapi(tokenized_corpus)
        return self._bm25
