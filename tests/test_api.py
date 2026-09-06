from langchain_core.messages import ToolMessage

from lims_agent.api.routes.chat import _extract_citations
from lims_agent.tools.sops import extract_sop_citations


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


def test_extract_sop_citations_supports_legacy_tool_message_content():
    content = (
        "[source: sop.pdf | document_id: sop-01 | title: Safety SOP | "
        "section: PPE | page 4]\nWear gloves and eye protection."
    )

    assert extract_sop_citations(content)[0]["pages"] == [4]
