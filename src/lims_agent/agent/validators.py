import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ResponseValidator:
    FORBIDDEN_PHRASES = [
        "i think",
        "probably",
        "likely",
        "maybe",
        "perhaps",
        "it seems",
        "i believe",
        "in my opinion",
        "based on my knowledge",
        "from what i know",
        "presumably",
        "supposedly",
        "might be",
        "could be",
        "should be around",
    ]

    @classmethod
    def extract_numbers(cls, text: str) -> set[str]:
        if not text:
            return set()
        numbers = re.findall(r"\d+\.?\d*", str(text))
        return set(numbers)

    @classmethod
    def check_forbidden_phrases(cls, response: str) -> list[str]:
        if not response:
            return []

        response_lower = response.lower()
        found = []

        for phrase in cls.FORBIDDEN_PHRASES:
            if phrase in response_lower:
                found.append(phrase)
                logger.warning("hallucination.forbidden_phrase phrase=%r", phrase)

        return found

    @classmethod
    def verify_numbers_grounded(
        cls, response: str, tool_outputs: list[str]
    ) -> tuple[bool, list[str]]:
        if not response:
            return True, []

        response_numbers = cls.extract_numbers(response)

        tool_numbers = set()
        for output in tool_outputs:
            tool_numbers.update(cls.extract_numbers(output))

        hallucinated = response_numbers - tool_numbers

        if hallucinated:
            logger.warning(
                "hallucination.ungrounded_numbers response_nums=%s tool_nums=%s hallucinated=%s",
                response_numbers,
                tool_numbers,
                hallucinated,
            )

        return len(hallucinated) == 0, list(hallucinated)

    @classmethod
    def validate_response(
        cls,
        response: str,
        tool_outputs: list[str] | None = None,
        strict_mode: bool = True,
    ) -> dict[str, Any]:
        violations = []
        hallucinated_numbers = []
        forbidden_phrases = []

        forbidden = cls.check_forbidden_phrases(response)
        if forbidden:
            violations.append("Contains uncertain language")
            forbidden_phrases = forbidden

        if tool_outputs:
            is_grounded, hallucinated = cls.verify_numbers_grounded(response, tool_outputs)
            if not is_grounded:
                violations.append("Contains numbers not in tool outputs")
                hallucinated_numbers = hallucinated

        is_valid = len(violations) == 0 or not strict_mode

        if not is_valid:
            logger.warning(
                "hallucination.validation_failed violations=%s",
                violations,
            )

        return {
            "is_valid": is_valid,
            "violations": violations,
            "hallucinated_numbers": hallucinated_numbers,
            "forbidden_phrases": forbidden_phrases,
        }
