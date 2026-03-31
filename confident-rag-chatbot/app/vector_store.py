"""FAISS vector database management for the RAG chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import FAISS
except ImportError as exc:  # pragma: no cover - import safety
    raise ImportError(
        "langchain-community is required for the FAISS vector store. "
        "Install dependencies from requirements.txt."
    ) from exc


class FAISSVectorStore:
    """Create, persist, load, and query a FAISS vector store."""

    def __init__(self, embeddings, index_path: Path) -> None:
        self.embeddings = embeddings
        self.index_path = Path(index_path)
        self.index = None

    def build(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build a vector store with no documents.")

        self.index = FAISS.from_documents(documents, self.embeddings)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Vector store has not been built yet.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_local(str(self.index_path))

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. Run the ingestion script first."
            )

        self.index = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.index is None:
            raise ValueError("Vector store is not loaded.")

        return self.index.similarity_search(query, k=k)
