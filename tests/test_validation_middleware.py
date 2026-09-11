from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from core.config import Settings
from middleware.validation import ValidationMiddleware


def _settings(**overrides) -> Settings:
    defaults = dict(
        langsmith_api_key="fake-key",
        block_uncertain_responses=True,
        langsmith_tracing=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class _StubGraph:
    """A LangChain Runnable so collect_runs() actually captures a run."""

    def __init__(self, response_text: str):
        self._runnable = RunnableLambda(self._respond)
        self._response_text = response_text

    async def _respond(self, _input_data: dict) -> dict:
        return {"messages": [AIMessage(content=self._response_text)]}

    async def ainvoke(self, input_data: dict, config: dict | None = None) -> dict:
        return await self._runnable.ainvoke(input_data, config=config)


class _ToolEvidenceGraph:
    async def ainvoke(self, input_data: dict, config: dict | None = None) -> dict:
        return {
            "messages": [
                ToolMessage(
                    content="A caustic substance has 2 < pH < 12.",
                    tool_call_id="call-1",
                ),
                AIMessage(content="The pH range for injectables is 2 < pH < 12."),
            ]
        }


def test_no_langsmith_client_when_tracing_enabled_but_key_blank():
    middleware = ValidationMiddleware(_settings(langsmith_tracing=True, langsmith_api_key=""))
    assert middleware._langsmith_client is None


@pytest.mark.asyncio
async def test_validation_reads_tool_evidence_before_final_answer():
    middleware = ValidationMiddleware(_settings(langsmith_tracing=False))
    invoke = middleware.wrap_agent(_ToolEvidenceGraph())

    result = await invoke(
        {"messages": [("user", "What is the pH range for injectables?")]},
        config={},
    )

    assert result["_validation"]["is_valid"] is False
    assert result["messages"][-1].content == (
        "I do not have that information in the authorized SOPs."
    )


@pytest.mark.asyncio
async def test_validated_invoke_works_without_langsmith_tracing():
    middleware = ValidationMiddleware(_settings(langsmith_tracing=False))
    assert middleware._langsmith_client is None

    invoke = middleware.wrap_agent(_StubGraph("The pH range is 7.0-7.4"))
    result = await invoke({"messages": []}, config={})

    assert result["messages"][-1].content == "The pH range is 7.0-7.4"


@pytest.mark.asyncio
async def test_validated_invoke_records_feedback_when_tracing_enabled(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr("middleware.validation.Client", lambda *a, **kw: fake_client)

    middleware = ValidationMiddleware(_settings(langsmith_tracing=True))
    invoke = middleware.wrap_agent(_StubGraph("The pH range is 7.0-7.4"))

    await invoke({"messages": []}, config={})

    fake_client.create_feedback.assert_called_once()
    _, kwargs = fake_client.create_feedback.call_args
    assert kwargs["key"] == "hallucination_check"
    assert kwargs["score"] is True


@pytest.mark.asyncio
async def test_validated_invoke_survives_langsmith_failure(monkeypatch):
    fake_client = MagicMock()
    fake_client.create_feedback.side_effect = RuntimeError("network down")
    monkeypatch.setattr("middleware.validation.Client", lambda *a, **kw: fake_client)

    middleware = ValidationMiddleware(_settings(langsmith_tracing=True))
    invoke = middleware.wrap_agent(_StubGraph("The pH range is 7.0-7.4"))

    result = await invoke({"messages": []}, config={})

    assert result["messages"][-1].content == "The pH range is 7.0-7.4"


@pytest.mark.asyncio
async def test_invalid_response_is_replaced_and_flagged():
    middleware = ValidationMiddleware(_settings(langsmith_tracing=False))
    invoke = middleware.wrap_agent(_StubGraph("I think the pH is probably 7.2"))

    result = await invoke({"messages": []}, config={})

    last = result["messages"][-1]
    assert last.content == "I do not have that information in the authorized SOPs."
    assert last.additional_kwargs["validation_failed"]["is_valid"] is False


def _build_real_agent_graph(response_text: str):
    """A real compiled StateGraph with a checkpointer, so persistence is genuine."""

    def node(_state: dict) -> dict:
        return {"messages": [AIMessage(content=response_text, id="ai-1")]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", node)
    workflow.add_edge(START, "agent")
    return workflow.compile(checkpointer=MemorySaver())


@pytest.mark.asyncio
async def test_invalid_response_is_corrected_in_checkpointed_state():
    agent_graph = _build_real_agent_graph("I think the pH is probably 7.2")
    middleware = ValidationMiddleware(_settings(langsmith_tracing=False))
    invoke = middleware.wrap_agent(agent_graph)
    config = {"configurable": {"thread_id": "thread-1"}}

    result = await invoke({"messages": []}, config=config)

    fallback = "I do not have that information in the authorized SOPs."
    assert result["messages"][-1].content == fallback

    state = await agent_graph.aget_state(config)
    persisted_last_message = state.values["messages"][-1]
    assert (
        persisted_last_message.content == "I do not have that information in the authorized SOPs."
    )
