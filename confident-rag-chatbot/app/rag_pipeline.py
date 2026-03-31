"""Core Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

from typing import List

import requests
from langchain_core.documents import Document

from app.config import (
    GROQ_API_BASE,
    GROQ_API_KEY,
    LLM_PROVIDER,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K_RESULTS,
    VECTOR_DB_PATH,
)
from app.embeddings import SentenceTransformerEmbeddings
from app.vector_store import FAISSVectorStore


class RAGPipeline:
    """Retrieve relevant context and generate grounded answers."""

    SYSTEM_PROMPT = (
        "You are an AI assistant for Confident Group. "
        "Answer only using the provided context. "
        "If the answer is not found in the context, say you do not know."
    )

    def __init__(self, embedding_model: str) -> None:
        self.embeddings = SentenceTransformerEmbeddings(embedding_model)
        self.vector_store = FAISSVectorStore(self.embeddings, VECTOR_DB_PATH)
        self.vector_store.load()

    def ask(self, question: str) -> str:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        documents = self.vector_store.similarity_search(question, k=TOP_K_RESULTS)
        context = self._format_context(documents)
        prompt = self._build_prompt(question=question.strip(), context=context)
        return self._generate_answer(prompt)

    def _format_context(self, documents: List[Document]) -> str:
        if not documents:
            return "No relevant context found."

        formatted_chunks = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            formatted_chunks.append(
                f"[Source {index}: {source}]\n{document.page_content.strip()}"
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
        if LLM_PROVIDER == "ollama":
            return self._call_ollama(prompt)

        return self._call_groq(prompt)

    def _call_groq(self, prompt: str) -> str:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or switch "
                "LLM_PROVIDER to 'ollama'."
            )

        response = requests.post(
            f"{GROQ_API_BASE.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            },
            timeout=60,
        )
        response.raise_for_status()

        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()

        payload = response.json()
        return payload["response"].strip()
