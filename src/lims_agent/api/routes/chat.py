from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends

from lims_agent.api.deps import get_agent_invoke
from lims_agent.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completion(
    payload: ChatRequest,
    agent_invoke: Callable[[dict, dict | None], Awaitable[dict]] = Depends(get_agent_invoke),
) -> ChatResponse:
    inputs = {"messages": [("user", payload.message)]}
    config = {"configurable": {"thread_id": payload.thread_id}}
    result = await agent_invoke(inputs, config)
    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=payload.thread_id,
    )
