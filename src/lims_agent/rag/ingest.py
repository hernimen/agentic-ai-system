import logging

from langchain_chroma import Chroma

from lims_agent.core.config import get_settings
from lims_agent.core.logging import configure_logging
from lims_agent.rag.loader import load_and_split_docs
from lims_agent.rag.retriever import build_embeddings

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    chunks = load_and_split_docs(settings)
    if not chunks:
        logger.warning("ingest.no_chunks — nothing to index")
        return

    Chroma.from_documents(
        documents=chunks,
        embedding=build_embeddings(settings),
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )
    logger.info("ingest.done chunks=%d collection=%s", len(chunks), settings.chroma_collection)


if __name__ == "__main__":
    main()
