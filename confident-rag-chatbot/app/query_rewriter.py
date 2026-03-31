"""Query rewriting utilities for retrieval-friendly user questions."""

from __future__ import annotations

from app.config import ENABLE_QUERY_REWRITE, QUERY_REWRITE_TIMEOUT_SECONDS
from app.llm_client import LLMClient


class QueryRewriter:
    """Rewrite vague questions into concise retrieval-oriented queries."""

    SYSTEM_PROMPT = (
        "You rewrite user questions for search and retrieval. "
        "Return one concise rewritten query that preserves the user's intent, "
        "adds missing company-specific context when helpful, and avoids commentary. "
        "If the original query is already specific, return it unchanged."
    )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def rewrite(self, query: str) -> str:
        cleaned_query = query.strip()
        if not cleaned_query:
            return query
        if not ENABLE_QUERY_REWRITE:
            return cleaned_query

        prompt = (
            "Rewrite the following user question into a concise retrieval-friendly query "
            "for Confident Group documents.\n\n"
            f"User question: {cleaned_query}\n\n"
            "Rewritten query:"
        )

        try:
            rewritten = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.0,
                timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
            ).strip()
        except Exception:
            return cleaned_query

        if not rewritten:
            return cleaned_query

        return " ".join(rewritten.splitlines()).strip() or cleaned_query
