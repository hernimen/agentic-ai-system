import logging

from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings

from lims_agent.core.config import Settings, get_settings
from lims_agent.rag.loader import load_and_split_docs
from lims_agent.rag.query_rewriter import QueryRewriter, build_query_rewriter
from lims_agent.rag.reranker import DocumentReranker, build_reranker

logger = logging.getLogger(__name__)


def build_embeddings(settings: Settings | None = None) -> OllamaEmbeddings:
    settings = settings or get_settings()
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)


def build_vector_store(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=build_embeddings(settings),
        persist_directory=settings.chroma_persist_dir,
    )


class QueryRewritingRetriever(BaseRetriever):
    """Wrapper retriever that rewrites queries before retrieval."""

    base_retriever: BaseRetriever
    rewriter: QueryRewriter

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Synchronous retrieval with query rewriting."""
        import asyncio

        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Async retrieval with query rewriting."""
        logger.info("rag.query_rewriting original=%r", query)

        try:
            query_variants = await self.rewriter.rewrite(query)
            logger.info("rag.query_rewriting variants=%d", len(query_variants))
        except Exception:
            logger.warning("rag.query_rewriting_failed falling_back_to_original", exc_info=True)
            query_variants = [query]

        all_docs: list[Document] = []
        for variant in query_variants:
            docs = await self.base_retriever.ainvoke(variant)
            all_docs.extend(docs)

        seen_content: set[str] = set()
        unique_docs: list[Document] = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)

        logger.info("rag.query_rewriting retrieved=%d unique=%d", len(all_docs), len(unique_docs))
        return unique_docs


class RerankingRetriever(BaseRetriever):
    """Wrapper retriever that re-ranks documents after retrieval."""

    base_retriever: BaseRetriever
    reranker: DocumentReranker

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Synchronous retrieval with re-ranking."""
        import asyncio

        return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Async retrieval with re-ranking."""
        docs = await self.base_retriever.ainvoke(query)

        if not docs:
            return docs

        try:
            ranked_docs = await self.reranker.rerank(docs, query)
            return ranked_docs
        except Exception:
            logger.warning("rag.reranking_failed returning_original", exc_info=True)
            return docs


def build_retriever(settings: Settings | None = None) -> BaseRetriever:
    settings = settings or get_settings()

    vector_retriever = build_vector_store(settings).as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": settings.retriever_k,
            "score_threshold": settings.retriever_score_threshold,
        },
    )

    bm25 = BM25Retriever.from_documents(load_and_split_docs(settings))
    bm25.k = settings.retriever_k

    logger.info(
        "retriever.built k=%d bm25_w=%.2f vector_w=%.2f",
        settings.retriever_k,
        settings.bm25_weight,
        settings.vector_weight,
    )

    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[settings.bm25_weight, settings.vector_weight],
    )

    retriever: BaseRetriever = ensemble
    if settings.query_rewriting_enabled:
        rewriter = build_query_rewriter(settings)
        retriever = QueryRewritingRetriever(base_retriever=retriever, rewriter=rewriter)
        logger.info(
            "retriever.query_rewriting_enabled strategy=%s variants=%d",
            settings.query_rewriting_strategy,
            settings.query_rewriting_num_variants,
        )

    if settings.reranking_enabled:
        reranker = build_reranker(settings)
        retriever = RerankingRetriever(base_retriever=retriever, reranker=reranker)
        logger.info(
            "retriever.reranking_enabled model=%s top_k=%d",
            settings.reranker_model,
            settings.reranker_top_k,
        )

    return retriever
