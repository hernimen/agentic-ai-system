import logging
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from lims_agent.core.config import Settings, get_settings
from lims_agent.llm import get_llm

logger = logging.getLogger(__name__)


class QueryRewriter(ABC):
    """Base class for query rewriting strategies."""

    @abstractmethod
    async def rewrite(self, query: str) -> list[str]:
        """
        Rewrite query into multiple variants.

        Args:
            query: Original user query

        Returns:
            List of query variants (includes original query)
        """
        pass


class MultiQueryRewriter(QueryRewriter):
    """LLM-based multi-query generation for semantic paraphrasing."""

    def __init__(self, llm: BaseChatModel, num_variants: int = 3, temperature: float = 0.3):
        self.llm = llm
        self.num_variants = num_variants
        self.temperature = temperature

    async def rewrite(self, query: str) -> list[str]:
        """Generate semantic paraphrases of the query using LLM."""
        system_prompt = SystemMessage(
            content=(
                "You are a query expansion assistant for a laboratory SOP search system. "
                f"Given a user query, generate {self.num_variants - 1} alternative phrasings that "
                "capture the same intent but use different vocabulary. Focus on:\n"
                "- Expanding abbreviations (SOP → Standard Operating Procedure)\n"
                "- Adding context (pH → pH range, pH limits, pH requirements)\n"
                "- Rephrasing questions into procedural terms\n"
                "- Including both specific (SOP-02) and general (oral solutions) terms\n\n"
                "Return only the alternative queries, one per line, without numbering or "
                "explanation."
            )
        )

        user_message = HumanMessage(content=query)

        try:
            response = await self.llm.ainvoke([system_prompt, user_message])

            variants = [line.strip() for line in response.content.split("\n") if line.strip()]
            variants = variants[: self.num_variants - 1]
            result = [query] + variants

            logger.info(
                "rag.query_rewriting strategy=multi_query original=%r variants=%d",
                query,
                len(result),
            )
            return result

        except Exception:
            logger.warning(
                "rag.query_rewriting_failed strategy=multi_query returning_original",
                exc_info=True,
            )
            return [query]


def build_query_rewriter(settings: Settings | None = None) -> QueryRewriter:
    """
    Factory function to create configured query rewriter.

    Args:
        settings: Application settings (defaults to get_settings())

    Returns:
        Configured MultiQueryRewriter instance
    """
    settings = settings or get_settings()
    llm = get_llm(settings)
    return MultiQueryRewriter(
        llm,
        num_variants=settings.query_rewriting_num_variants,
        temperature=settings.query_rewriting_temperature,
    )
