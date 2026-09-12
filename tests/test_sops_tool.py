from langchain_core.documents import Document

from tools.sops import _document_matches_identifier, _document_matches_subject, make_get_sop_rules


def test_get_sop_rules_returns_concatenated_docs(stub_retriever):
    tool = make_get_sop_rules(stub_retriever)
    result = tool.invoke({"query": "pH injectables"})
    assert "6.8 and 7.4" in result


def test_get_sop_rules_handles_empty(stub_retriever):
    stub_retriever.docs = []
    tool = make_get_sop_rules(stub_retriever)
    assert tool.invoke({"query": "unknown"}) == "No SOP rules found for the given query."


def test_get_sop_rules_limits_evidence_documents(stub_retriever):
    stub_retriever.docs = [
        Document(page_content=f"Evidence {index}", metadata={"source_name": f"sop-{index}.pdf"})
        for index in range(3)
    ]

    content = make_get_sop_rules(stub_retriever).invoke({"query": "pH"})

    assert content.count("[source:") == 3


def test_get_sop_rules_prefers_documents_matching_named_instrument(stub_retriever):
    stub_retriever.docs = [
        Document(
            page_content="Autoclave sterilization procedure.",
            metadata={"source_name": "autoclave.pdf"},
        ),
        Document(
            page_content="Shaker operation procedure.", metadata={"source_name": "shaker.pdf"}
        ),
        Document(
            page_content="General biosafety requirements.", metadata={"source_name": "safety.pdf"}
        ),
    ]

    content = make_get_sop_rules(stub_retriever).invoke(
        {"query": "autoclave operation procedure safety"}
    )

    assert "autoclave.pdf" in content
    assert "shaker.pdf" not in content
    assert "safety.pdf" not in content


def test_get_sop_rules_prioritizes_explicit_sop_identifier(stub_retriever):
    stub_retriever.docs = [
        Document(
            page_content="SOP-02 storage temperature is 2-8 C.",
            metadata={"document_id": "SOP-02", "source_name": "sop-02.pdf"},
        ),
        Document(
            page_content="SOP-07 incubation time is 24 hours.",
            metadata={"document_id": "SOP-07", "source_name": "sop-07.pdf"},
        ),
    ]

    content = make_get_sop_rules(stub_retriever).invoke({"query": "SOP-02 storage temperature"})

    assert "sop-02.pdf" in content
    assert "sop-07.pdf" not in content


def test_sop_identifier_formats_match_same_document():
    doc = Document(page_content="Storage requirements.", metadata={"document_id": "sop-02"})

    assert _document_matches_identifier(doc, "sop-2")


def test_sop_identifier_does_not_match_longer_identifier():
    doc = Document(page_content="Storage requirements.", metadata={"document_id": "sop-020"})

    assert not _document_matches_identifier(doc, "sop-02")


def test_subject_matching_uses_complete_tokens():
    doc = Document(page_content="Laboratory safety requirements.")

    assert not _document_matches_subject(doc, "lab")
