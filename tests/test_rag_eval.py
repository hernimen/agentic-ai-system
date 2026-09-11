from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from lims_agent.eval.benchmark import (
    EvaluationCase,
    _build_ragas_rows,
    check_benchmark_thresholds,
    evaluate_case,
    run_benchmark_suite,
)
from lims_agent.eval.dataset import build_default_benchmark_cases
from lims_agent.rag.ingest import chunk_batches
from lims_agent.rag.loader import enrich_document_metadata, split_document_into_sections
from lims_agent.rag.query_rewriter import build_query_retriever
from lims_agent.rag.retriever import summarize_retrieval_results
from lims_agent.tools.sops import make_get_sop_rules


def test_build_default_benchmark_cases_returns_realistic_sop_questions():
    cases = build_default_benchmark_cases()

    assert len(cases) >= 5
    assert any("ph" in case.question.lower() for case in cases)
    assert any("ppe" in case.question.lower() for case in cases)
    assert any("temperature" in case.question.lower() for case in cases)
    assert all(case.context for case in cases)


def test_query_retriever_uses_multi_query_and_keeps_original(monkeypatch):
    class RecordingRetriever(BaseRetriever):
        queries: list[str] = Field(default_factory=list)

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
            self.queries.append(query)
            return [Document(page_content="shared evidence"), Document(page_content=query)]

    base_retriever = RecordingRetriever()
    fake_llm = FakeListChatModel(responses=["pH range\npH limits"])
    monkeypatch.setattr("lims_agent.rag.query_rewriter.get_llm", lambda settings: fake_llm)

    retriever = build_query_retriever(base_retriever)
    results = retriever.invoke("pH injectables")

    assert isinstance(retriever, MultiQueryRetriever)
    assert "pH injectables" in base_retriever.queries
    assert len(results) == 4
    assert [doc.page_content for doc in results].count("shared evidence") == 1


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

    doc = Document(page_content="sample text", metadata={"page": 3})
    enriched = enrich_document_metadata(doc, source_dir, source_file)

    assert enriched.metadata["source"].endswith("sop_general_lab_2016-01.pdf")
    assert enriched.metadata["source_path"].endswith("sop_general_lab_2016-01.pdf")
    assert enriched.metadata["document_id"] == "sop_general_lab_2016-01"
    assert enriched.metadata["doc_type"] == "sops"
    assert enriched.metadata["source_category"] == "sop"
    assert enriched.metadata["title"] == "sop general lab 2016 01"
    assert enriched.metadata["page_start"] == 3
    assert enriched.metadata["page_end"] == 3


def test_run_benchmark_suite_returns_summary_metrics():
    cases = build_default_benchmark_cases()
    report = run_benchmark_suite(cases)

    assert report["total_cases"] == len(cases)
    assert report["summary"]["pass_rate"] >= 0.0
    assert report["summary"]["passed"] >= 0
    assert "grounded_rate" in report["summary"]
    assert "avg_source_coverage" in report["summary"]


def test_build_ragas_rows_uses_retrieved_context_text():
    cases = [
        EvaluationCase(
            question="What is the pH range?",
            expected_answer="The pH range is 6.8 to 7.4.",
            retrieved_sources=["SOP-02"],
            context="The authorized pH range is 6.8 to 7.4.",
            answer="The pH range is 6.8 to 7.4.",
        )
    ]

    rows = _build_ragas_rows(cases)

    assert rows[0]["contexts"] == ["The authorized pH range is 6.8 to 7.4."]


def test_check_benchmark_thresholds_raises_when_quality_is_too_low():
    cases = [
        EvaluationCase(
            question="What is the pH range?",
            expected_answer="The pH range is 6.8 to 7.4.",
            expected_sources=["SOP-02"],
            retrieved_sources=["SOP-99"],
            answer="The pH range is 7.0 to 7.8.",
        )
    ]

    try:
        check_benchmark_thresholds(
            cases,
            min_pass_rate=0.9,
            min_grounded_rate=0.9,
            min_retrieval_hit_rate=0.9,
        )
    except ValueError as exc:
        assert "pass_rate" in str(exc).lower()
        return

    raise AssertionError("Expected benchmark threshold check to fail for poor quality")


def test_split_document_into_sections_keeps_section_metadata():
    doc = Document(
        page_content=(
            "Purpose\n\nThis SOP describes the procedure.\n\n"
            "Safety\n\nWear gloves and goggles before handling samples."
        ),
        metadata={"title": "sample sop", "source_name": "sample.pdf"},
    )

    sections = split_document_into_sections(doc)

    assert len(sections) >= 2
    assert all("section_title" in section.metadata for section in sections)
    assert sections[0].metadata["title"] == "sample sop"
    assert sections[0].metadata["source_name"] == "sample.pdf"


def test_chunk_batches_limits_embedding_batch_size():
    chunks = list(range(7))

    assert list(chunk_batches(chunks, 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_summarize_retrieval_results_tracks_source_metadata():
    docs = [
        Document(
            page_content="The pH range is between 6.8 and 7.4.",
            metadata={
                "source": "/data/sops/sop_general_lab_2016-01.pdf",
                "source_name": "sop_general_lab_2016-01.pdf",
                "document_id": "sop_general_lab_2016-01",
                "title": "sop general lab 2016 01",
                "section_title": "pH requirements",
                "page_start": 14,
                "page_end": 14,
            },
        )
    ]

    summary = summarize_retrieval_results(docs)

    assert summary["total_docs"] == 1
    assert summary["documents"][0]["source_name"] == "sop_general_lab_2016-01.pdf"
    assert summary["documents"][0]["document_id"] == "sop_general_lab_2016-01"
    assert summary["documents"][0]["section_title"] == "pH requirements"
    assert summary["documents"][0]["page_start"] == 14


def test_make_get_sop_rules_includes_explicit_evidence_metadata():
    class FakeRetriever:
        def invoke(self, query):
            return [
                Document(
                    page_content="The pH range is between 6.8 and 7.4.",
                    metadata={
                        "source": "/data/sops/sop_general_lab_2016-01.pdf",
                        "source_name": "sop_general_lab_2016-01.pdf",
                        "document_id": "sop_general_lab_2016-01",
                        "title": "sop general lab 2016 01",
                        "section_title": "pH requirements",
                        "page_start": 14,
                        "page_end": 14,
                    },
                )
            ]

    tool = make_get_sop_rules(FakeRetriever())
    result = tool.invoke("pH range")

    assert "sop_general_lab_2016-01" in result
    assert "pH requirements" in result
    assert "page 14" in result
    assert "sop general lab 2016 01" in result
