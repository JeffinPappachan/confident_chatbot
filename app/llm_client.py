"""Shared LLM client utilities for provider-backed text generation."""

from __future__ import annotations

from typing import Any, Dict

import requests

from app.config import (
    GROQ_API_BASE,
    GROQ_API_KEY,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
)


class LLMClient:
    """Simple provider-aware chat/generation client."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or LLM_PROVIDER).strip().lower()

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        timeout: int | None = None,
    ) -> str:
        timeout_seconds = timeout or LLM_TIMEOUT_SECONDS

        if self.provider == "ollama":
            prompt = f"{system_prompt}\n\n{user_prompt}".strip()
            return self._call_ollama(prompt, timeout_seconds)
        if self.provider == "openai":
            return self._call_openai_compatible(
                api_base=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                timeout=timeout_seconds,
                temperature=temperature,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                provider_name="OpenAI",
            )
        if self.provider == "groq":
            return self._call_openai_compatible(
                api_base=GROQ_API_BASE,
                api_key=GROQ_API_KEY,
                timeout=timeout_seconds,
                temperature=temperature,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                provider_name="Groq",
            )

        raise RuntimeError(
            f"Unsupported LLM_PROVIDER '{self.provider}'. "
            "Use one of: groq, openai, ollama."
        )

    def _call_openai_compatible(
        self,
        *,
        api_base: str,
        api_key: str,
        timeout: int,
        temperature: float,
        system_prompt: str,
        user_prompt: str,
        provider_name: str,
    ) -> str:
        if not api_key:
            raise RuntimeError(
                f"{provider_name.upper()}_API_KEY is not set. "
                f"Add it to your environment or switch LLM_PROVIDER from '{self.provider}'."
            )

        response = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def _call_ollama(self, prompt: str, timeout: int) -> str:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        return payload["response"].strip()
