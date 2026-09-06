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
            metadata = doc.metadata or {}
            source = metadata.get("source", "unknown")
            source_name = metadata.get("source_name") or metadata.get("source", "unknown")
            document_id = metadata.get("document_id") or "unknown-document"
            title = metadata.get("title") or "untitled"
            section_title = metadata.get("section_title") or "section"
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")

            if page_start is not None and page_end is not None and page_start == page_end:
                page_info = f"page {page_start}"
            elif page_start is not None and page_end is not None:
                page_info = f"pages {page_start}-{page_end}"
            elif page_start is not None:
                page_info = f"page {page_start}"
            else:
                page_info = "page unknown"

            label = (
                "[source: "
                f"{source_name} | document_id: {document_id} | title: {title} | "
                f"section: {section_title} | {page_info}]"
            )
            entries.append(f"{label}\n{doc.page_content}")
        return "\n\n".join(entries)

    return get_sop_rules
