from collections.abc import Awaitable, Callable

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph


def get_agent_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.agent_graph


def get_agent_invoke(request: Request) -> Callable[[dict, dict | None], Awaitable[dict]]:
    return request.app.state.agent_invoke
