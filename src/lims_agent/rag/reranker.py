import asyncio
import logging
from abc import ABC, abstractmethod

from langchain_core.documents import Document

from lims_agent.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class DocumentReranker(ABC):
    """Base class for document re-ranking strategies."""

    @abstractmethod
    async def rerank(self, documents: list[Document], query: str) -> list[Document]:
        """
        Rerank documents by relevance to the query.

        Args:
            documents: List of retrieved documents
            query: Original user query

        Returns:
            Documents sorted by relevance (highest first), limited to top_k
        """
        pass


class CrossEncoderReranker(DocumentReranker):
    """Local cross-encoder re-ranking using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", top_k: int = 5):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    def _load_model(self):
        """Lazy load model to avoid startup cost."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("rag.reranking loading_model model=%s", self.model_name)
                self._model = CrossEncoder(self.model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._model

    async def rerank(self, documents: list[Document], query: str) -> list[Document]:
        """Score query-document pairs and return top-k sorted by relevance."""
        if not documents:
            return documents

        try:
            model = self._load_model()

            pairs = [(query, doc.page_content) for doc in documents]
            scores = await asyncio.to_thread(model.predict, pairs)

            scored_docs = list(zip(documents, scores, strict=True))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            top_docs = [doc for doc, score in scored_docs[: self.top_k]]

            logger.info(
                "rag.reranking strategy=cross_encoder total=%d top_k=%d "
                "scores_range=(%.3f,%.3f) model=%s",
                len(documents),
                len(top_docs),
                float(scores.min()),
                float(scores.max()),
                self.model_name,
            )
            return top_docs

        except Exception:
            logger.warning(
                "rag.reranking_failed strategy=cross_encoder returning_unranked",
                exc_info=True,
            )
            return documents[: self.top_k]


def build_reranker(settings: Settings | None = None) -> DocumentReranker:
    """
    Factory function to create configured reranker.

    Args:
        settings: Application settings (defaults to get_settings())

    Returns:
        Configured DocumentReranker instance
    """
    settings = settings or get_settings()
    return CrossEncoderReranker(
        model_name=settings.reranker_model,
        top_k=settings.reranker_top_k,
    )
