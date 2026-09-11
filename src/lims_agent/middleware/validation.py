import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tracers.context import collect_runs
from langsmith import Client

from lims_agent.agent.validators import ResponseValidator
from lims_agent.core.config import Settings

logger = logging.getLogger(__name__)


class ValidationMiddleware:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.validator = ResponseValidator()
        self._langsmith_client = (
            Client() if settings.langsmith_tracing and settings.langsmith_api_key else None
        )

    def wrap_agent(
        self, agent_graph: Any
    ) -> Callable[[dict[str, Any], dict[str, Any] | None], Awaitable[dict[str, Any]]]:
        async def validated_invoke(
            input_data: dict[str, Any], config: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            if self._langsmith_client is None:
                result = await agent_graph.ainvoke(input_data, config=config)
                traced_runs = []
            else:
                with collect_runs() as run_collector:
                    result = await agent_graph.ainvoke(input_data, config=config)
                traced_runs = run_collector.traced_runs

            last_message = result["messages"][-1]
            tool_outputs = self._extract_tool_outputs(result["messages"])
            query = self._extract_query(input_data)

            validation = self.validator.validate_response(
                response=str(last_message.content),
                query=query,
                tool_outputs=tool_outputs if tool_outputs else None,
                strict_mode=self.settings.block_uncertain_responses,
            )

            if self._langsmith_client and traced_runs:
                self._record_validation_feedback(str(traced_runs[0].id), validation)

            if not validation["is_valid"]:
                logger.error("validation.failed violations=%s", validation["violations"])
                fallback = "I do not have that information in the authorized SOPs."
                last_message.content = fallback
                if hasattr(last_message, "additional_kwargs"):
                    last_message.additional_kwargs["validation_failed"] = validation

                if config is not None and hasattr(agent_graph, "aupdate_state"):
                    await agent_graph.aupdate_state(
                        config,
                        {"messages": [last_message]},
                        as_node="agent",
                    )

            result["_validation"] = validation
            return result

        return validated_invoke

    def _record_validation_feedback(self, run_id: str, validation: dict[str, Any]) -> None:
        if self._langsmith_client is None:
            return
        try:
            self._langsmith_client.create_feedback(
                run_id=run_id,
                key="hallucination_check",
                score=validation["is_valid"],
                value={"violations": validation["violations"]},
            )
        except Exception:
            logger.warning("langsmith.feedback_failed run_id=%s", run_id, exc_info=True)

    def _extract_tool_outputs(self, messages: list[Any]) -> list[str]:
        tool_outputs: list[str] = []
        final_ai_seen = False
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_outputs.append(str(msg.content))
            elif hasattr(msg, "type") and msg.type == "ai":
                if final_ai_seen:
                    break
                final_ai_seen = True
        return tool_outputs

    def _extract_query(self, input_data: dict[str, Any]) -> str:
        messages = input_data.get("messages", [])
        if not messages:
            return ""

        latest_message = messages[-1]
        if isinstance(latest_message, tuple):
            return str(latest_message[1])
        return str(getattr(latest_message, "content", ""))
