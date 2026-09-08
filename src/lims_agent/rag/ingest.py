import logging
from collections.abc import Iterator

from langchain_chroma import Chroma

from lims_agent.core.config import get_settings
from lims_agent.core.logging import configure_logging
from lims_agent.rag.loader import load_and_split_docs
from lims_agent.rag.retriever import build_embeddings

logger = logging.getLogger(__name__)
INGEST_BATCH_SIZE = 32


def chunk_batches[DocumentType](
    items: list[DocumentType], batch_size: int
) -> Iterator[list[DocumentType]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    chunks = load_and_split_docs(settings)
    if not chunks:
        logger.warning("ingest.no_chunks — nothing to index")
        return

    vector_store = Chroma(
        embedding_function=build_embeddings(settings),
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )
    vector_store.reset_collection()
    for batch in chunk_batches(chunks, INGEST_BATCH_SIZE):
        vector_store.add_documents(batch)
    logger.info("ingest.done chunks=%d collection=%s", len(chunks), settings.chroma_collection)


if __name__ == "__main__":
    main()
