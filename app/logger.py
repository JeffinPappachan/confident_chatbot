"""Non-blocking JSON query logging for chatbot requests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import LOG_PATH


def _serialize_documents(documents: list[Document]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for document in documents:
        serialized.append(
            {
                "chunk_id": document.metadata.get("chunk_id"),
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "preview": document.page_content[:200],
            }
        )
    return serialized


class QueryLogger:
    """Persist query execution details without interrupting chatbot responses."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = Path(log_path or LOG_PATH)

    def log(
        self,
        *,
        original_query: str,
        rewritten_query: str,
        retrieved_chunks: list[Document],
        reranked_chunks: list[Document],
        response_time_ms: float,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "retrieved_chunks": _serialize_documents(retrieved_chunks),
            "reranked_chunks": _serialize_documents(reranked_chunks),
            "llm_response_time_ms": round(response_time_ms, 2),
        }

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logs = []
            if self.log_path.exists():
                try:
                    existing = json.loads(self.log_path.read_text(encoding="utf-8"))
                    if isinstance(existing, list):
                        logs = existing
                except Exception:
                    logs = []

            logs.append(payload)
            self.log_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
        except Exception:
            return
