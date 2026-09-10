from langchain_core.messages import AIMessage, ToolMessage

from lims_agent.api.deps import get_agent_invoke
from lims_agent.api.routes.chat import _extract_citations


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
        {
            "source_name": "sop_general_lab.pdf",
            "document_id": "sop-01",
            "title": "general lab sop",
            "section": "pH requirements",
            "pages": [14],
            "excerpt": "The pH range is between 6.8 and 7.4.",
        }
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
