from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from core.config import Settings, get_settings
from llm import get_llm


def build_query_retriever(
    base_retriever: BaseRetriever,
    settings: Settings | None = None,
) -> BaseRetriever:
    """
    Factory function to create configured query rewriter.

    Args:
        settings: Application settings (defaults to get_settings())

    Returns:
        Configured LangChain MultiQueryRetriever instance.
    """
    settings = settings or get_settings()
    llm = get_llm(settings)
    prompt = PromptTemplate.from_template(
        """You are a query expansion assistant for a laboratory SOP search system.
Given the user query below, generate {variant_count} alternative phrasings that capture the same
intent but use different vocabulary. Focus on:
- Expanding abbreviations (SOP -> Standard Operating Procedure)
- Adding context (pH -> pH range, pH limits, pH requirements)
- Rephrasing questions into procedural terms
- Including both specific and general terms from the query

Return only the alternative queries, one per line, without numbering or explanation.

Original query: {question}"""
    ).partial(variant_count=str(max(settings.query_rewriting_num_variants - 1, 1)))
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=prompt,
        include_original=True,
    )
