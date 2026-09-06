from lims_agent.eval.benchmark import EvaluationCase, evaluate_case


def test_evaluate_case_marks_grounded_answer_as_pass():
    case = EvaluationCase(
        question="What is the pH range for injectables?",
        expected_answer="The pH range for injectables is between 6.8 and 7.4.",
        expected_sources=["SOP-02", "SOP-09"],
        retrieved_sources=["SOP-02", "SOP-09"],
        answer="The pH range for injectables is between 6.8 and 7.4.",
    )

    result = evaluate_case(case)

    assert result["passed"] is True
    assert result["grounded"] is True
    assert result["retrieval_hit"] is True
    assert result["source_coverage"] == 1.0


def test_evaluate_case_flags_ungrounded_answer():
    case = EvaluationCase(
        question="What is the pH range for injectables?",
        expected_answer="The pH range for injectables is between 6.8 and 7.4.",
        expected_sources=["SOP-02"],
        retrieved_sources=["SOP-99"],
        answer="The pH range for injectables is between 7.0 and 7.8.",
    )

    result = evaluate_case(case)

    assert result["passed"] is False
    assert result["grounded"] is False
    assert result["retrieval_hit"] is False
