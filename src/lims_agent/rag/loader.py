import logging
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lims_agent.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def enrich_document_metadata(
    doc: Document,
    source_dir: Path,
    source_file: Path,
) -> Document:
    source_name = source_file.name
    stem = source_file.stem
    doc_type = source_dir.name
    title = stem.replace("_", " ").replace("-", " ").strip().lower()

    metadata = dict(doc.metadata or {})
    metadata.update(
        {
            "source": str(source_file),
            "source_name": source_name,
            "document_id": stem,
            "doc_type": doc_type,
            "title": title,
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

        for source_file, doc in zip(source_files, loaded_docs, strict=False):
            docs.append(enrich_document_metadata(doc, source_dir, source_file))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info("sops.loaded raw_docs=%d chunks=%d", len(docs), len(chunks))
    return chunks
