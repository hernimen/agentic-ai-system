import os

from langchain_core.messages import AIMessage, ToolMessage

from lims_agent.api.deps import get_agent_invoke
from lims_agent.api.main import configure_langsmith
from lims_agent.api.routes.chat import _extract_citations
from lims_agent.api.schemas import Citation
from lims_agent.core.config import Settings


def test_configure_langsmith_sets_environment(monkeypatch):
    for variable in (
        "LANGSMITH_TRACING",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.setenv(variable, "original")

    settings = Settings(
        langsmith_tracing=True,
        langsmith_api_key="test-key",
        langsmith_endpoint="https://example.test",
        langsmith_project="test-project",
    )

    configure_langsmith(settings)

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://example.test"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"


def test_chat_completions_smoke(client):
    response = client.post(
        "/api/v1/chat/completions",
        json={"message": "hello", "thread_id": "t1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "response": "stub response",
        "thread_id": "t1",
        "citations": [],
        "grounded": None,
    }


def test_chat_completions_hides_citations_for_invalid_response(client):
    async def invalid_invoke(inputs: dict, config: dict | None = None) -> dict:
        return {
            "messages": [
                ToolMessage(
                    content="A caustic substance has 2 < pH < 12.",
                    tool_call_id="call-1",
                    artifact=[{"source_name": "irrelevant.pdf"}],
                ),
                AIMessage(content="I do not have that information in the authorized SOPs."),
            ],
            "_validation": {"is_valid": False},
        }

    client.app.dependency_overrides[get_agent_invoke] = lambda: invalid_invoke

    response = client.post(
        "/api/v1/chat/completions",
        json={"message": "pH for injectables", "thread_id": "t1"},
    )

    assert response.json()["grounded"] is False
    assert response.json()["citations"] == []


def test_extract_citations_preserves_structured_source_metadata():
    messages = [
        ToolMessage(
            content="The pH range is between 6.8 and 7.4.",
            tool_call_id="call-1",
            artifact=[
                {
                    "source_name": "sop_general_lab.pdf",
                    "document_id": "sop-01",
                    "title": "general lab sop",
                    "section": "pH requirements",
                    "pages": [14],
                    "excerpt": "The pH range is between 6.8 and 7.4.",
                }
            ],
        )
    ]

    assert _extract_citations(messages) == [
        Citation(
            source_name="sop_general_lab.pdf",
            document_id="sop-01",
            title="general lab sop",
            section="pH requirements",
            pages=[14],
            excerpt="The pH range is between 6.8 and 7.4.",
        )
    ]


def test_extract_citations_does_not_fallback_when_artifact_is_empty():
    messages = [
        ToolMessage(
            content=(
                "[source: legacy.pdf | document_id: legacy-01 | title: Legacy | "
                "section: Old | page 2]\nLegacy evidence."
            ),
            tool_call_id="call-1",
            artifact=[],
        )
    ]

    assert _extract_citations(messages) == []


def test_health_route_reports_liveness(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_route_reports_dependency_status(client, monkeypatch):
    monkeypatch.setattr("lims_agent.api.routes.health._check_chroma_ready", lambda: True)
    monkeypatch.setattr("lims_agent.api.routes.health._check_ollama_ready", lambda: True)

    response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["chroma"] == "ok"
    assert payload["checks"]["ollama"] == "ok"


def test_health_ready_route_returns_503_when_dependency_fails(client, monkeypatch):
    monkeypatch.setattr("lims_agent.api.routes.health._check_chroma_ready", lambda: True)

    def fail_ollama() -> bool:
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr("lims_agent.api.routes.health._check_ollama_ready", fail_ollama)

    response = client.get("/health/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "error"
    assert "ollama" in payload["checks"]
