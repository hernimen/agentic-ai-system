import logging
import re
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lims_agent.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _normalize_title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().lower()


def _infer_source_category(source_dir: Path, source_file: Path) -> str:
    source_dir_name = source_dir.name.lower()
    source_name = source_file.name.lower()

    if "sop" in source_dir_name or "sop" in source_name:
        return "sop"
    if "equipment" in source_dir_name or "manual" in source_name:
        return "equipment_manual"
    return "other"


def split_document_into_sections(doc: Document) -> list[Document]:
    text = doc.page_content.strip()
    if not text:
        return [doc]

    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    if len(blocks) <= 1:
        return [doc]

    sections: list[Document] = []
    for idx, block in enumerate(blocks):
        section_title = block.splitlines()[0].strip() if block else "section"
        section_text = block.strip()
        metadata = dict(doc.metadata or {})
        metadata.update(
            {
                "section_index": idx,
                "section_title": section_title,
                "section_text": section_text,
            }
        )
        sections.append(Document(page_content=section_text, metadata=metadata))

    return sections


def enrich_document_metadata(
    doc: Document,
    source_dir: Path,
    source_file: Path,
) -> Document:
    source_name = source_file.name
    stem = source_file.stem
    doc_type = source_dir.name
    title = _normalize_title_from_path(source_file)

    metadata = dict(doc.metadata or {})
    page_start = metadata.get("page") or metadata.get("page_start") or 1
    page_end = metadata.get("page_end") or page_start

    metadata.update(
        {
            "source": str(source_file),
            "source_path": str(source_file),
            "source_name": source_name,
            "source_category": _infer_source_category(source_dir, source_file),
            "document_id": stem,
            "doc_type": doc_type,
            "title": title,
            "page_start": int(page_start),
            "page_end": int(page_end),
        }
    )
    doc.metadata = metadata
    return doc


def load_and_split_docs(settings: Settings | None = None) -> list[Document]:
    settings = settings or get_settings()

    docs: list[Document] = []
    for raw_dir in settings.sops_source_dirs:
        source_dir = Path(raw_dir)

        if not source_dir.exists():
            logger.warning("sops.source_dir_missing path=%s", source_dir)
            continue

        source_files = sorted(source_dir.glob("**/*.pdf"))
        loaded_docs = DirectoryLoader(
            str(source_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        ).load()

        source_by_name = {source_file.name: source_file for source_file in source_files}
        for doc in loaded_docs:
            doc_source = Path(str(doc.metadata.get("source", "")))
            source_file = source_by_name.get(doc_source.name)
            if source_file is None:
                fallback_name = doc_source.name or Path(str(doc.metadata.get("file_path", ""))).name
                source_file = source_by_name.get(fallback_name)
            if source_file is None:
                logger.warning(
                    "sops.unmatched_document source_dir=%s metadata=%s",
                    source_dir,
                    doc.metadata,
                )
                continue

            enriched = enrich_document_metadata(doc, source_dir, source_file)
            docs.extend(split_document_into_sections(enriched))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info("sops.loaded raw_docs=%d chunks=%d", len(docs), len(chunks))
    return chunks
