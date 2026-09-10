import subprocess
import sys

from langchain_core.documents import Document

from lims_agent.rag import reranker as reranker_module
from lims_agent.rag.reranker import CrossEncoderReranker


def test_reranker_module_imports_without_sentence_transformers():
    script = """
import importlib.abc
import sys


class BlockSentenceTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sentence_transformers" or fullname.startswith("sentence_transformers."):
            raise ModuleNotFoundError("sentence-transformers unavailable")
        return None


sys.meta_path.insert(0, BlockSentenceTransformers())
sys.modules.pop("sentence_transformers", None)
import lims_agent.rag.reranker
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


async def test_cross_encoder_rerank_offloads_prediction_to_worker(monkeypatch):
    class Scores(list[float]):
        def min(self):
            return min(self)

        def max(self):
            return max(self)

    class FakeModel:
        def predict(self, pairs):
            calls.append(pairs)
            return Scores([0.1, 0.9])

    calls = []
    worker_calls = []

    async def fake_to_thread(function, *args):
        worker_calls.append(function)
        return function(*args)

    reranker = CrossEncoderReranker(top_k=1)
    monkeypatch.setattr(reranker, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(reranker_module.asyncio, "to_thread", fake_to_thread)

    documents = [Document(page_content="low relevance"), Document(page_content="high relevance")]
    result = await reranker.rerank(documents, "laboratory query")

    assert result == [documents[1]]
    assert worker_calls == [worker_calls[0]]
    assert calls == [
        [("laboratory query", "low relevance"), ("laboratory query", "high relevance")]
    ]
