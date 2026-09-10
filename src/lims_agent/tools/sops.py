import logging
import re

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

MAX_EVIDENCE_DOCUMENTS = 8
_QUERY_TERM = re.compile(r"[a-z0-9]{3,}")
_QUERY_CONTEXT_TERMS = {"operation", "procedure", "safety", "parameters", "parameter"}
_SOP_IDENTIFIER = re.compile(r"\bsop[- ]?(\d+)\b", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    """Lowercase and collapse whitespace so comparisons are deterministic."""
    return " ".join(value.lower().split())


def _extract_sop_identifier(query: str) -> str | None:
    """Return an SOP code like 'sop-02' from the user query when present."""
    match = _SOP_IDENTIFIER.search(query)
    if match is None:
        return None
    return f"sop-{match.group(1)}"


def _extract_subject_term(query: str) -> str | None:
    """Find the main subject in a query while ignoring generic context words."""
    terms = _QUERY_TERM.findall(_normalize_text(query))
    for term in terms:
        if term not in _QUERY_CONTEXT_TERMS:
            return term
    return None


def _document_matches_identifier(doc: object, identifier: str) -> bool:
    """Check whether the document text or ID clearly references the requested SOP."""
    metadata = getattr(doc, "metadata", {}) or {}
    haystack = f"{metadata.get('document_id', '')} {getattr(doc, 'page_content', '')}"
    normalized_haystack = _normalize_text(haystack).replace(" ", "")
    return identifier in normalized_haystack


def _document_matches_subject(doc: object, subject_term: str) -> bool:
    """Check whether the document section is about the queried subject."""
    metadata = getattr(doc, "metadata", {}) or {}
    haystack = f"{getattr(doc, 'page_content', '')} {metadata.get('section_title', '')}"
    return subject_term in _normalize_text(haystack)


def _deduplicate_documents(docs: list[object]) -> list[object]:
    """Keep the first copy of each content block while preserving order."""
    seen_content: set[str] = set()
    unique_docs: list[object] = []
    for doc in docs:
        page_content = getattr(doc, "page_content", "")
        if page_content in seen_content:
            continue
        seen_content.add(page_content)
        unique_docs.append(doc)
    return unique_docs


def _format_page_info(page_start: int | None, page_end: int | None) -> str:
    """Create a compact page-description string for evidence labels."""
    if page_start is not None and page_end is not None and page_start == page_end:
        return f"page {page_start}"
    if page_start is not None and page_end is not None:
        return f"pages {page_start}-{page_end}"
    if page_start is not None:
        return f"page {page_start}"
    return "page unknown"


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

        identifier = _extract_sop_identifier(query)
        if identifier:
            matching_docs = [doc for doc in docs if _document_matches_identifier(doc, identifier)]
            if matching_docs:
                docs = matching_docs
        else:
            subject_term = _extract_subject_term(query)
            if subject_term:
                matching_docs = [
                    doc for doc in docs if _document_matches_subject(doc, subject_term)
                ]
                if matching_docs:
                    docs = matching_docs

        unique_docs = _deduplicate_documents(docs)

        entries: list[str] = []
        citations: list[dict] = []
        for doc in unique_docs[:MAX_EVIDENCE_DOCUMENTS]:
            metadata = doc.metadata or {}
            source_name = metadata.get("source_name") or metadata.get("source", "unknown")
            document_id = metadata.get("document_id") or "unknown-document"
            title = metadata.get("title") or "untitled"
            section_title = metadata.get("section_title") or "section"
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            page_info = _format_page_info(page_start, page_end)

            label = (
                "[source: "
                f"{source_name} | document_id: {document_id} | title: {title} | "
                f"section: {section_title} | {page_info}]"
            )
            entries.append(f"{label}\n{doc.page_content}")

            pages: list[int] = []
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
