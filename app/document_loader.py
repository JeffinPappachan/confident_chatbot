"""Utilities for loading company documents into LangChain document objects."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError as exc:  # pragma: no cover - import safety
    raise ImportError(
        "langchain-community is required for document loading. "
        "Install dependencies from requirements.txt."
    ) from exc


def normalize_metadata(metadata: dict) -> dict:
    """Normalize metadata shape across PDF and website documents."""

    normalized = dict(metadata)
    source_value = str(normalized.get("source", "unknown"))
    if source_value.lower().endswith(".pdf"):
        normalized["source"] = Path(source_value).name
    else:
        normalized["source"] = source_value

    raw_page = normalized.get("page", 0)
    try:
        page_number = int(raw_page)
        if normalized.get("type") == "pdf" and page_number >= 0:
            page_number += 1
        elif page_number <= 0:
            page_number = 1
    except (TypeError, ValueError):
        page_number = 1

    normalized["page"] = page_number
    return normalized


def load_pdf_documents(data_dir: Path) -> List[Document]:
    """Load all PDF files from the given directory."""

    documents: List[Document] = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {data_dir}")

    pdf_files = sorted(data_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in {data_dir}. Add company PDFs before ingesting."
        )

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        loaded_documents = loader.load()
        for document in loaded_documents:
            document.metadata = normalize_metadata(
                {
                    **(document.metadata or {}),
                    "source": pdf_file.name,
                    "page": (document.metadata or {}).get("page", 0),
                    "type": "pdf",
                }
            )
        documents.extend(loaded_documents)

    return documents


def load_website_documents(urls: Sequence[str], timeout: int = 20) -> List[Document]:
    """Fetch and parse website pages into LangChain documents."""

    documents: List[Document] = []
    headers = {"User-Agent": "ConfidentGroupRAGBot/1.0"}

    for url in urls:
        cleaned_url = url.strip()
        if not cleaned_url:
            continue

        response = requests.get(cleaned_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata=normalize_metadata(
                    {"source": cleaned_url, "page": 1, "type": "website"}
                ),
            )
        )

    return documents


def load_documents(
    data_dir: Path, website_urls: Iterable[str] | None = None
) -> List[Document]:
    """Load PDF documents and optionally website content."""

    documents = load_pdf_documents(data_dir)

    if website_urls:
        documents.extend(load_website_documents(list(website_urls)))

    return documents
