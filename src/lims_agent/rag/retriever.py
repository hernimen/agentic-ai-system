import logging
import re
from typing import cast

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from pydantic import PrivateAttr
from rank_bm25 import BM25Okapi

from lims_agent.core.config import Settings, get_settings
from lims_agent.rag.loader import load_and_split_docs
from lims_agent.rag.query_rewriter import build_query_retriever
from lims_agent.rag.reranker import DocumentReranker, build_reranker

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """Lightweight BM25 retriever backed by rank_bm25 instead of langchain-community."""

    documents: list[Document]
    k: int = 4
    _bm25: BM25Okapi = PrivateAttr()
    _document_tokens: list[list[str]] = PrivateAttr()

    def __init__(self, documents: list[Document], *, k: int = 4):
        super().__init__(documents=documents, k=k)  # type: ignore[call-arg]
        self._document_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._document_tokens)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    @classmethod
    def from_documents(cls, documents: list[Document], *, k: int = 4) -> "BM25Retriever":
        return cls(documents, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        query_token_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self.documents)), key=lambda index: scores[index], reverse=True
        )[: self.k]
        return [
            self.documents[index]
            for index in ranked_indices
            if query_token_set.intersection(self._document_tokens[index])
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: (
            AsyncCallbackManagerForRetrieverRun | CallbackManagerForRetrieverRun | None
        ) = None,
    ) -> list[Document]:
        sync_run_manager = cast(CallbackManagerForRetrieverRun | None, run_manager)
        if isinstance(run_manager, AsyncCallbackManagerForRetrieverRun):
            sync_run_manager = None
        return self._get_relevant_documents(query, run_manager=sync_run_manager)


def summarize_retrieval_results(docs: list[Document]) -> dict:
    """Return a compact record of retrieved docs for observability and debugging."""
    summarized = []
    for doc in docs:
        metadata = doc.metadata or {}
        summarized.append(
            {
                "source": metadata.get("source", "unknown"),
                "source_name": metadata.get("source_name") or metadata.get("source", "unknown"),
                "document_id": metadata.get("document_id") or "unknown-document",
                "title": metadata.get("title") or "untitled",
                "section_title": metadata.get("section_title") or "section",
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "content_preview": doc.page_content[:200],
            }
        )

    return {
        "total_docs": len(docs),
        "documents": summarized,
    }


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


class RerankingRetriever(BaseRetriever):
    """Wrapper retriever that re-ranks documents after retrieval."""

    base_retriever: BaseRetriever
    reranker: DocumentReranker

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Synchronous retrieval with re-ranking."""
        import asyncio

        return asyncio.run(
            self._aget_relevant_documents(
                query,
                run_manager=run_manager if run_manager is not None else None,
            )
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: (
            AsyncCallbackManagerForRetrieverRun | CallbackManagerForRetrieverRun | None
        ) = None,
    ) -> list[Document]:
        """Async retrieval with re-ranking."""
        docs = await self.base_retriever.ainvoke(query)

        if not docs:
            return docs

        try:
            ranked_docs = await self.reranker.rerank(docs, query)
            logger.info("rag.reranking summary=%s", summarize_retrieval_results(ranked_docs))
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
        retriever = build_query_retriever(ensemble, settings)
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
