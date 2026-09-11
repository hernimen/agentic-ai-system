from langchain_core.documents import Document

from rag.retriever import BM25Retriever


def test_bm25_retriever_constructs_and_returns_matching_documents():
    matching = Document(page_content="SOP-01 requires pH between 6.8 and 7.4.")
    unrelated = Document(page_content="Store samples at 2 to 8 degrees Celsius.")
    retriever = BM25Retriever.from_documents([matching, unrelated], k=2)

    assert retriever.invoke("pH") == [matching]
