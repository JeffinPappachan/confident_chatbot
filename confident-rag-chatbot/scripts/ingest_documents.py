"""Ingest Confident Group documents into a FAISS vector index."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL, VECTOR_DB_PATH
from app.document_loader import load_documents
from app.embeddings import SentenceTransformerEmbeddings
from app.vector_store import FAISSVectorStore


def main() -> None:
    website_urls = [
        url.strip() for url in os.getenv("WEBSITE_URLS", "").split(",") if url.strip()
    ]

    print("Loading documents...")
    documents = load_documents(DATA_DIR, website_urls=website_urls)

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were generated from the provided documents.")

    print("Generating embeddings and building FAISS index...")
    embeddings = SentenceTransformerEmbeddings(EMBEDDING_MODEL)
    vector_store = FAISSVectorStore(embeddings, VECTOR_DB_PATH)
    vector_store.build(chunks)
    vector_store.save()

    print(f"Ingestion completed successfully. Index saved to: {VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()
