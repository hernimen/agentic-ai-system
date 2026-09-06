import logging

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

    def wrap_agent(self, agent_graph):
        async def validated_invoke(input_data: dict, config: dict | None = None) -> dict:
            with collect_runs() as run_collector:
                result = await agent_graph.ainvoke(input_data, config=config)

            last_message = result["messages"][-1]
            tool_outputs = self._extract_tool_outputs(result["messages"])

            validation = self.validator.validate_response(
                response=str(last_message.content),
                tool_outputs=tool_outputs if tool_outputs else None,
                strict_mode=self.settings.block_uncertain_responses,
            )

            if self._langsmith_client and run_collector.traced_runs:
                self._record_validation_feedback(run_collector.traced_runs[0].id, validation)

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

    def _record_validation_feedback(self, run_id, validation: dict) -> None:
        try:
            self._langsmith_client.create_feedback(
                run_id=run_id,
                key="hallucination_check",
                score=validation["is_valid"],
                value={"violations": validation["violations"]},
            )
        except Exception:
            logger.warning("langsmith.feedback_failed run_id=%s", run_id, exc_info=True)

    def _extract_tool_outputs(self, messages: list) -> list[str]:
        tool_outputs = []
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_outputs.append(str(msg.content))
            elif hasattr(msg, "type") and msg.type == "ai":
                break
        return tool_outputs
