from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Depends

from lims_agent.api.deps import get_agent_invoke
from lims_agent.api.schemas import ChatRequest, ChatResponse, Citation

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _extract_citations(messages: list[Any]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[tuple[str, str, str, tuple[int, ...], str]] = set()

    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue

        artifact = getattr(message, "artifact", None)
        if artifact is None:
            continue
        for citation in artifact:
            normalized = Citation.model_validate(citation)
            citation_key = (
                normalized.source_name,
                normalized.document_id,
                normalized.section,
                tuple(normalized.pages),
                normalized.excerpt,
            )
            if citation_key not in seen:
                seen.add(citation_key)
                citations.append(normalized)

    return citations


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completion(
    payload: ChatRequest,
    agent_invoke: Callable[[dict, dict | None], Awaitable[dict]] = Depends(get_agent_invoke),
) -> ChatResponse:
    inputs = {"messages": [("user", payload.message)]}
    config = {"configurable": {"thread_id": payload.thread_id}}
    result = await agent_invoke(inputs, config)
    messages = result["messages"]
    validation = result.get("_validation", {})
    citations = _extract_citations(messages) if validation.get("is_valid") is not False else []
    return ChatResponse(
        response=messages[-1].content,
        thread_id=payload.thread_id,
        citations=citations,
        grounded=validation.get("is_valid"),
    )
