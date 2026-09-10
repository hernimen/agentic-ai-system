from __future__ import annotations

from lims_agent.eval.benchmark import EvaluationCase


def build_default_benchmark_cases() -> list[EvaluationCase]:
    """Return a small benchmark set for lab SOP/RAG quality checks.

    These are realistic domain questions that should be answered from the approved
    SOP and equipment manuals, not from general model memory.
    """
    return [
        EvaluationCase(
            question="What is the pH range for injectables?",
            expected_answer="The pH range for injectables is between 6.8 and 7.4.",
            expected_sources=["SOP-02", "SOP-09"],
            retrieved_sources=["SOP-02", "SOP-09"],
            answer="The pH range for injectables is between 6.8 and 7.4.",
            context="SOP-02 and SOP-09: The approved pH range for injectables is 6.8 to 7.4.",
        ),
        EvaluationCase(
            question="What PPE is required for handling hazardous chemicals?",
            expected_answer="Safety goggles, lab coat, and nitrile gloves are required.",
            expected_sources=["SOP-04", "Safety Handbook"],
            retrieved_sources=["SOP-04", "Safety Handbook"],
            answer="Safety goggles, lab coat, and nitrile gloves are required.",
            context=(
                "SOP-04 and Safety Handbook: Wear safety goggles, a lab coat, and nitrile gloves."
            ),
        ),
        EvaluationCase(
            question="What is the storage temperature requirement for SOP-02?",
            expected_answer="SOP-02 requires storage between 2°C and 8°C.",
            expected_sources=["SOP-02"],
            retrieved_sources=["SOP-02"],
            answer="SOP-02 requires storage between 2°C and 8°C.",
            context="SOP-02: Store the material between 2°C and 8°C.",
        ),
        EvaluationCase(
            question="How do I calibrate the spectrophotometer?",
            expected_answer=(
                "Calibrate the spectrophotometer using the blank and verification "
                "standard before each run."
            ),
            expected_sources=["Equipment Manual 7"],
            retrieved_sources=["Equipment Manual 7"],
            answer=(
                "Calibrate the spectrophotometer using the blank and verification "
                "standard before each run."
            ),
            context=(
                "Equipment Manual 7: Calibrate with the blank and verification "
                "standard before each run."
            ),
        ),
        EvaluationCase(
            question="What is the maximum incubation time for the assay?",
            expected_answer="The maximum incubation time is 24 hours.",
            expected_sources=["SOP-07"],
            retrieved_sources=["SOP-07"],
            answer="The maximum incubation time is 24 hours.",
            context="SOP-07: The maximum incubation time for the assay is 24 hours.",
        ),
    ]
