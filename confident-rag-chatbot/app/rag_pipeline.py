"""Core Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List

from langchain_core.documents import Document

from app.config import EMBEDDING_MODEL, MAX_CONTEXT_TOKENS, VECTOR_DB_PATH
from app.embeddings import SentenceTransformerEmbeddings
from app.hybrid_retriever import HybridRetriever
from app.llm_client import LLMClient
from app.logger import QueryLogger
from app.query_rewriter import QueryRewriter
from app.reranker import Reranker
from app.vector_store import FAISSVectorStore


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict[str, int | str]]


class RAGPipeline:
    """Retrieve relevant context and generate grounded answers."""

    SYSTEM_PROMPT = (
        "You are an AI assistant for Confident Group. "
        "Answer the user's question using only the provided context. "
        "If the answer is not contained in the context, say that the information is not available."
    )

    def __init__(self, embedding_model: str = EMBEDDING_MODEL) -> None:
        self.embeddings = SentenceTransformerEmbeddings(embedding_model)
        self.vector_store = FAISSVectorStore(self.embeddings, VECTOR_DB_PATH)
        self.vector_store.load()
        self.llm_client = LLMClient()
        self.query_rewriter = QueryRewriter(self.llm_client)
        self.hybrid_retriever = HybridRetriever(self.vector_store)
        self.reranker = Reranker()
        self.query_logger = QueryLogger()

    def ask(self, question: str) -> RAGResponse:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        original_query = question.strip()
        rewritten_query = self.query_rewriter.rewrite(original_query)
        retrieved_documents = self.hybrid_retriever.retrieve(rewritten_query)
        reranked_documents = self.reranker.rerank(rewritten_query, retrieved_documents)
        guarded_documents = self._apply_context_window_guard(reranked_documents)
        context = self._format_context(guarded_documents)
        prompt = self._build_prompt(question=original_query, context=context)

        started_at = perf_counter()
        answer = self._generate_answer(prompt)
        response_time_ms = (perf_counter() - started_at) * 1000
        sources = self._extract_sources(guarded_documents)

        self.query_logger.log(
            original_query=original_query,
            rewritten_query=rewritten_query,
            retrieved_chunks=retrieved_documents,
            reranked_chunks=guarded_documents,
            response_time_ms=response_time_ms,
        )

        return RAGResponse(answer=answer, sources=sources)

    def _format_context(self, documents: List[Document]) -> str:
        if not documents:
            return "No relevant context found."

        formatted_chunks = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", "unknown")
            formatted_chunks.append(
                f"[Source {index}: {source} (page {page})]\n{document.page_content.strip()}"
            )

        return "\n\n".join(formatted_chunks)

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "Use the context below to answer the question about Confident Group.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _generate_answer(self, prompt: str) -> str:
        return self.llm_client.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.1,
        )

    def _apply_context_window_guard(self, documents: List[Document]) -> List[Document]:
        selected_documents: list[Document] = []
        current_tokens = 0

        for document in documents:
            estimated_tokens = self._estimate_tokens(document.page_content)
            if selected_documents and current_tokens + estimated_tokens > MAX_CONTEXT_TOKENS:
                break
            selected_documents.append(document)
            current_tokens += estimated_tokens

        return selected_documents or documents[:1]

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()) * 2)

    def _extract_sources(self, documents: List[Document]) -> list[dict[str, int | str]]:
        sources: list[dict[str, int | str]] = []
        seen: set[tuple[str, int | str]] = set()

        for document in documents:
            source = str(document.metadata.get("source", "unknown"))
            page = document.metadata.get("page", "unknown")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"document": source, "page": page})

        return sources
