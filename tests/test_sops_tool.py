from langchain_core.documents import Document

from lims_agent.tools.sops import make_get_sop_rules


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

    assert content.count("[source:") == 2
