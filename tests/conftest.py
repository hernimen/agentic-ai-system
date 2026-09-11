from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from api.deps import get_agent_graph, get_agent_invoke
from api.main import create_app


class _StubRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return self.docs


@pytest.fixture
def stub_retriever() -> BaseRetriever:
    return _StubRetriever(
        docs=[Document(page_content="SOP-01: pH for injectables must be between 6.8 and 7.4.")]
    )


class _StubGraph:
    async def ainvoke(self, inputs: dict, config: dict) -> dict:
        class _M:
            content = "stub response"

        return {"messages": [_M()]}


async def _stub_invoke(inputs: dict, config: dict | None = None) -> dict:
    class _M:
        content = "stub response"

    return {"messages": [_M()]}


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.router.lifespan_context = None  # skip real startup (no Ollama, no Chroma) in tests
    app.dependency_overrides[get_agent_graph] = lambda: _StubGraph()
    app.dependency_overrides[get_agent_invoke] = lambda: _stub_invoke
    yield TestClient(app)
    app.dependency_overrides.clear()
