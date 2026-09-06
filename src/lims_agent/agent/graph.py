from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lims_agent.agent.prompts import SYSTEM_PROMPT
from lims_agent.core.config import Settings, get_settings
from lims_agent.llm import get_llm


def build_agent_graph(
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    settings = settings or get_settings()
    llm = get_llm(settings)
    llm_with_tools = llm.bind_tools(list(tools))

    def call_model(state: MessagesState) -> dict:
        response = llm_with_tools.invoke([SYSTEM_PROMPT, *state["messages"]])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(list(tools)))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)
