from langchain_core.documents import Document

from lims_agent.eval.benchmark import EvaluationCase, evaluate_case
from lims_agent.eval.dataset import build_default_benchmark_cases
from lims_agent.rag.loader import enrich_document_metadata


def test_build_default_benchmark_cases_returns_realistic_sop_questions():
    cases = build_default_benchmark_cases()

    assert len(cases) >= 5
    assert any("ph" in case.question.lower() for case in cases)
    assert any("ppe" in case.question.lower() for case in cases)
    assert any("temperature" in case.question.lower() for case in cases)


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


def test_enrich_document_metadata_includes_source_fields(tmp_path):
    source_dir = tmp_path / "sops"
    source_dir.mkdir()
    source_file = source_dir / "sop_general_lab_2016-01.pdf"
    source_file.write_bytes(b"test")

    doc = Document(page_content="sample text", metadata={})
    enriched = enrich_document_metadata(doc, source_dir, source_file)

    assert enriched.metadata["source"].endswith("sop_general_lab_2016-01.pdf")
    assert enriched.metadata["document_id"] == "sop_general_lab_2016-01"
    assert enriched.metadata["doc_type"] == "sops"
    assert enriched.metadata["title"] == "sop general lab 2016 01"
