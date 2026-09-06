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


def load_and_split_docs(settings: Settings | None = None) -> list[Document]:
    settings = settings or get_settings()

    docs: list[Document] = []
    for raw_dir in settings.sops_source_dirs:
        source_dir = Path(raw_dir)

        if not source_dir.exists():
            logger.warning("sops.source_dir_missing path=%s", source_dir)
            continue

        docs.extend(
            DirectoryLoader(str(source_dir), glob="**/*.pdf", loader_cls=PyPDFLoader).load()
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info("sops.loaded raw_docs=%d chunks=%d", len(docs), len(chunks))
    return chunks
