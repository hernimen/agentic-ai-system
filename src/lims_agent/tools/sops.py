import logging

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)


def make_get_sop_rules(retriever: BaseRetriever) -> BaseTool:
    @tool
    def get_sop_rules(query: str) -> str:
        """
        Retrieve official Standard Operating Procedure (SOP) rules, regulatory limits,
        or pH ranges using a semantic query (e.g., 'pH limits for injectables').
        Use this tool when you need to verify compliance or search laboratory manuals.
        """
        logger.info("rag.query query=%r", query)
        docs = retriever.invoke(query)
        if not docs:
            return "No SOP rules found for the given query."

        seen_content: set[str] = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content in seen_content:
                continue
            seen_content.add(doc.page_content)
            unique_docs.append(doc)

        entries = []
        for doc in unique_docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            label = (
                f"[source: {source}, page {page}]" if page is not None else f"[source: {source}]"
            )
            entries.append(f"{label}\n{doc.page_content}")
        return "\n\n".join(entries)

    return get_sop_rules
