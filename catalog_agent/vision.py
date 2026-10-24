"""Ollama multimodal client (local HTTP API)."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)


class VisionModelError(RuntimeError):
    """The model server could not produce a response after retries."""


class VisionClient(Protocol):
    model: str

    def chat(self, messages: list[dict[str, Any]], json_schema: dict | None = None) -> str:
        """Send chat messages (images attached as base64 in `images`) and return the raw text reply."""


class OllamaVisionClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2-vision",
        timeout: float = 300.0,
        http_retries: int = 3,
        backoff_seconds: float = 2.0,
        temperature: float = 0.4,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.http_retries = max(1, http_retries)
        self.backoff_seconds = backoff_seconds
        self.temperature = temperature
        self._client = httpx.Client(base_url=host.rstrip("/"), timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def check(self) -> None:
        """Raise VisionModelError if the server is unreachable or the model isn't pulled."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisionModelError(f"Cannot reach Ollama at {self._client.base_url}: {exc}") from exc
        names = {m.get("name", "") for m in response.json().get("models", [])}
        if not any(name == self.model or name.split(":")[0] == self.model for name in names):
            raise VisionModelError(f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}")

    def chat(self, messages: list[dict[str, Any]], json_schema: dict | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        last_error: Exception | None = None
        for attempt in range(1, self.http_retries + 1):
            try:
                response = self._client.post("/api/chat", json=payload)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server error {response.status_code}: {response.text[:200]}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()["message"]["content"]
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                # 4xx (bad model name, bad payload) won't fix itself; fail fast.
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise VisionModelError(f"Ollama rejected the request: {exc.response.text[:300]}") from exc
                last_error = exc
                log.warning("Ollama request failed (attempt %d/%d): %s", attempt, self.http_retries, exc)
                if attempt < self.http_retries:
                    time.sleep(self.backoff_seconds * 2 ** (attempt - 1))
            except (KeyError, ValueError) as exc:
                raise VisionModelError(f"Unexpected Ollama response shape: {exc}") from exc
        raise VisionModelError(f"Ollama unavailable after {self.http_retries} attempts: {last_error}")
