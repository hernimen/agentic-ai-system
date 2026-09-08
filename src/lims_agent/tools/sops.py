import logging
import re

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

MAX_EVIDENCE_DOCUMENTS = 2

_EVIDENCE_LABEL = re.compile(
    r"\[source: (?P<source>[^|]+) \| document_id: (?P<document_id>[^|]+) \| "
    r"title: (?P<title>[^|]+) \| section: (?P<section>[^|]+) \| "
    r"(?P<page_info>pages? [^\]]+)\]\n(?P<excerpt>.*?)(?=\n\n\[source:|\Z)",
    re.DOTALL,
)


def extract_sop_citations(content: str) -> list[dict]:
    citations = []
    for match in _EVIDENCE_LABEL.finditer(content):
        citations.append(
            {
                "source_name": match.group("source").strip(),
                "document_id": match.group("document_id").strip(),
                "title": match.group("title").strip(),
                "section": match.group("section").strip(),
                "pages": [int(value) for value in re.findall(r"\d+", match.group("page_info"))],
                "excerpt": match.group("excerpt").strip(),
            }
        )
    return citations


def make_get_sop_rules(retriever: BaseRetriever) -> BaseTool:
    @tool(response_format="content_and_artifact")
    def get_sop_rules(query: str) -> tuple[str, list[dict]]:
        """
        Retrieve official Standard Operating Procedure (SOP) rules, regulatory limits,
        or pH ranges using a semantic query (e.g., 'pH limits for injectables').
        Use this tool when you need to verify compliance or search laboratory manuals.
        """
        logger.info("rag.query query=%r", query)
        docs = retriever.invoke(query)
        if not docs:
            return "No SOP rules found for the given query.", []

        seen_content: set[str] = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content in seen_content:
                continue
            seen_content.add(doc.page_content)
            unique_docs.append(doc)

        entries = []
        citations = []
        for doc in unique_docs[:MAX_EVIDENCE_DOCUMENTS]:
            metadata = doc.metadata or {}
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
            pages = []
            if page_start is not None:
                pages.append(int(page_start))
            if page_end is not None and page_end != page_start:
                pages.append(int(page_end))
            citations.append(
                {
                    "source_name": source_name,
                    "document_id": document_id,
                    "title": title,
                    "section": section_title,
                    "pages": pages,
                    "excerpt": doc.page_content,
                }
            )
        return "\n\n".join(entries), citations

    return get_sop_rules
