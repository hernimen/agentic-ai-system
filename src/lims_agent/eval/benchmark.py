from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationCase:
    question: str
    expected_answer: str
    expected_sources: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    answer: str = ""
    context: str = ""


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _source_coverage(expected: list[str], retrieved: list[str]) -> float:
    if not expected:
        return 1.0
    expected_set = {s.lower() for s in expected}
    retrieved_set = {s.lower() for s in retrieved}
    hits = len(expected_set & retrieved_set)
    return hits / len(expected_set)


def evaluate_case(case: EvaluationCase) -> dict[str, Any]:
    normalized_answer = _normalize(case.answer)
    normalized_expected = _normalize(case.expected_answer)

    answer_matches = normalized_answer == normalized_expected
    retrieval_hit = bool(case.retrieved_sources) and _source_coverage(
        case.expected_sources,
        case.retrieved_sources,
    ) >= 0.5
    source_coverage = _source_coverage(case.expected_sources, case.retrieved_sources)

    grounded = answer_matches and retrieval_hit
    passed = grounded and source_coverage >= 0.5

    return {
        "question": case.question,
        "passed": passed,
        "grounded": grounded,
        "retrieval_hit": retrieval_hit,
        "source_coverage": source_coverage,
        "answer_matches_expected": answer_matches,
        "expected_sources": case.expected_sources,
        "retrieved_sources": case.retrieved_sources,
    }
