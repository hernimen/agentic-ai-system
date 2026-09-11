from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="groq")
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.0
    ollama_base_url: str = Field(default="http://localhost:11434")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    embedding_model: str = "nomic-embed-text"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "lims_sops"

    sops_source_dirs: list[str] = Field(default=["./data/sops", "./data/equipment_manuals"])
    chunk_size: int = 500
    chunk_overlap: int = 50

    retriever_k: int = 2
    retriever_score_threshold: float = 0.0
    bm25_weight: float = 0.5
    vector_weight: float = 0.5

    # Query rewriting settings
    query_rewriting_enabled: bool = Field(default=True)
    query_rewriting_strategy: str = Field(default="multi_query")
    query_rewriting_num_variants: int = Field(default=3)

    # Re-ranking settings
    reranking_enabled: bool = Field(default=True)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    reranker_top_k: int = Field(default=5)

    database_url: str = Field(default="sqlite:///./data/lims_agent.db")

    block_uncertain_responses: bool = Field(default=True)

    log_level: str = Field(default="INFO")
    api_title: str = "POC Agentic SOP and AI assistant"
    api_version: str = "0.1.0"

    langsmith_tracing: bool = Field(default=True, validation_alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias="LANGSMITH_ENDPOINT",
    )
    langsmith_api_key: str = Field(default="", validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="lims-sop-agent", validation_alias="LANGSMITH_PROJECT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
