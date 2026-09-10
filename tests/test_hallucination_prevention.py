from lims_agent.agent.prompts import SYSTEM_PROMPT
from lims_agent.agent.validators import ResponseValidator


class TestResponseValidator:
    def test_prompt_routes_equipment_questions_to_sop_retrieval(self):
        prompt = SYSTEM_PROMPT.content.lower()

        assert "laboratory equipment" in prompt
        assert "any named laboratory instrument" in prompt
        assert "descriptive or broad equipment questions" in prompt
        assert "do not substitute one parameter" in prompt
        assert "operation, procedure, safety, and parameters" in prompt

    def test_extract_numbers(self):
        text = "pH range 7.0-7.4 at 25°C for 2 hours"
        numbers = ResponseValidator.extract_numbers(text)
        assert "7" in numbers or "7.0" in numbers
        assert "7.4" in numbers
        assert "25" in numbers
        assert "2" in numbers

    def test_detect_forbidden_phrases(self):
        bad_responses = [
            "The pH is probably 7.2",
            "I think the storage temperature is 25°C",
            "It seems like the incubation time is 24 hours",
            "Maybe the pH should be around 6.5",
        ]

        for response in bad_responses:
            phrases = ResponseValidator.check_forbidden_phrases(response)
            assert len(phrases) > 0, f"Should detect forbidden phrase in: {response}"

        good_responses = [
            "The pH range is 7.0-7.4",
            "Storage temperature: 2-8°C",
            "Incubation time is 24 hours",
        ]

        for response in good_responses:
            phrases = ResponseValidator.check_forbidden_phrases(response)
            assert len(phrases) == 0, f"False positive in: {response}"

    def test_number_grounding_valid(self):
        response = "The pH range is 7.0-7.4"
        tool_outputs = ["SOP-02: pH must be maintained between 7.0 and 7.4"]

        is_valid, hallucinated = ResponseValidator.verify_numbers_grounded(response, tool_outputs)

        assert is_valid, "Valid numbers should pass"
        assert len(hallucinated) == 0

    def test_number_grounding_normalizes_decimal_formatting(self):
        response = "The target pH is 7.0."
        tool_outputs = ["SOP-02: The target pH is 7.00."]

        is_valid, hallucinated = ResponseValidator.verify_numbers_grounded(response, tool_outputs)

        assert is_valid
        assert hallucinated == []

    def test_number_grounding_invalid(self):
        response = "The pH range is 6.5-7.5"
        tool_outputs = ["SOP-02: pH must be maintained between 7.0 and 7.4"]

        is_valid, hallucinated = ResponseValidator.verify_numbers_grounded(response, tool_outputs)

        assert not is_valid, "Hallucinated numbers should fail"
        assert len(hallucinated) > 0

    def test_validate_response_strict_mode(self):
        result = ResponseValidator.validate_response(
            response="The pH is probably 7.2",
            tool_outputs=["pH: 7.2"],
            strict_mode=True,
        )
        assert not result["is_valid"]
        assert len(result["forbidden_phrases"]) > 0

        result = ResponseValidator.validate_response(
            response="The temperature is 30°C",
            tool_outputs=["Temperature: 25°C"],
            strict_mode=True,
        )
        assert not result["is_valid"]
        assert len(result["hallucinated_numbers"]) > 0

        result = ResponseValidator.validate_response(
            response="The temperature is 25°C",
            tool_outputs=["Temperature: 25°C"],
            strict_mode=True,
        )
        assert result["is_valid"]
        assert len(result["violations"]) == 0

    def test_no_tool_outputs_skips_number_check(self):
        result = ResponseValidator.validate_response(
            response="The temperature is 30°C",
            tool_outputs=None,
            strict_mode=True,
        )
        assert result["is_valid"]

    def test_rejects_numbers_when_query_concept_is_missing_from_evidence(self):
        result = ResponseValidator.validate_response(
            response="The pH range for injectables is 2 < pH < 12.",
            query="What is the acceptable pH range for injectables?",
            tool_outputs=["A caustic substance has 2 < pH < 12."],
            strict_mode=True,
        )

        assert not result["is_valid"]
        assert "Query concepts are not grounded in tool outputs" in result["violations"]

    def test_empty_response(self):
        result = ResponseValidator.validate_response(
            response="",
            tool_outputs=[],
            strict_mode=True,
        )
        assert result["is_valid"]
