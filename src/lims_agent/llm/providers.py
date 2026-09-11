import ssl
from abc import ABC, abstractmethod
from typing import Any

import httpx
import truststore
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr


class LLMProvider(ABC):
    @abstractmethod
    def get_model(self) -> BaseChatModel:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class GroqProvider(LLMProvider):
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.kwargs = kwargs

    def get_model(self) -> BaseChatModel:
        try:
            from langchain_groq import ChatGroq  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "langchain-groq not installed. Install with: uv sync --group groq"
            ) from e

        ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return ChatGroq(
            model=self.model,
            temperature=self.temperature,
            api_key=SecretStr(self.api_key) if self.api_key is not None else None,
            http_client=httpx.Client(verify=ssl_context),
            http_async_client=httpx.AsyncClient(verify=ssl_context),
            **self.kwargs,
        )

    @property
    def name(self) -> str:
        return f"groq/{self.model}"
