import logging
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)


class ResponseValidator:
    QUERY_STOPWORDS = {
        "a",
        "an",
        "and",
        "for",
        "are",
        "how",
        "is",
        "of",
        "the",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "from",
        "does",
        "acceptable",
        "required",
        "maximum",
        "como",
        "cual",
        "de",
        "del",
        "el",
        "en",
        "es",
        "la",
        "las",
        "los",
        "para",
        "por",
        "que",
        "segun",
        "una",
        "uno",
        "y",
        "o",
    }
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
        normalized_numbers = set()
        for number in numbers:
            try:
                normalized_numbers.add(format(Decimal(number).normalize(), "f"))
            except InvalidOperation:
                normalized_numbers.add(number)
        return normalized_numbers

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
    def verify_query_grounded(
        cls, response: str, query: str, tool_outputs: list[str]
    ) -> tuple[bool, list[str]]:
        """Check that the evidence covers the question's meaningful terms."""
        if not query or not tool_outputs:
            return True, []

        def tokens(value: str) -> set[str]:
            normalized = unicodedata.normalize("NFKD", value.lower())
            normalized = "".join(char for char in normalized if not unicodedata.combining(char))
            return {
                token
                for token in re.findall(r"[a-z0-9]{2,}", normalized)
                if token not in cls.QUERY_STOPWORDS
            }

        query_terms = tokens(query)
        response_terms = tokens(response)
        evidence_terms = tokens(" ".join(tool_outputs))
        claimed_query_terms = query_terms & response_terms
        missing_terms = sorted(claimed_query_terms - evidence_terms)

        if missing_terms:
            logger.warning(
                "hallucination.ungrounded_query_terms claimed_terms=%s missing_terms=%s",
                claimed_query_terms,
                missing_terms,
            )
            return False, missing_terms

        return True, []

    @classmethod
    def validate_response(
        cls,
        response: str,
        tool_outputs: list[str] | None = None,
        strict_mode: bool = True,
        query: str = "",
    ) -> dict[str, Any]:
        violations = []
        hallucinated_numbers = []
        missing_query_terms = []
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

            query_is_grounded, missing_query_terms = cls.verify_query_grounded(
                response, query, tool_outputs
            )
            if not query_is_grounded:
                violations.append("Query concepts are not grounded in tool outputs")

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
            "missing_query_terms": missing_query_terms,
            "forbidden_phrases": forbidden_phrases,
        }
