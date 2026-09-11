import logging

from langchain_core.language_models import BaseChatModel

from core.config import Settings, get_settings
from llm.providers import GroqProvider

logger = logging.getLogger(__name__)


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    provider_name = settings.llm_provider.lower()

    if provider_name != "groq":
        raise ValueError(f"Unknown LLM provider: {provider_name}. Only 'groq' is supported.")

    provider = GroqProvider(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )

    logger.info(f"llm.initialized provider={provider.name}")
    return provider.get_model()
