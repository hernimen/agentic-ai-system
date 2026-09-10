from langchain_chroma import Chroma
from langchain_core.documents import Document


class DeterministicEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_chroma_persists_document_content_and_provenance(tmp_path):
    persist_directory = str(tmp_path / "chroma")
    collection_name = "lims_persistence_test"
    embeddings = DeterministicEmbeddings()
    metadata = {
        "source_name": "sop_storage.pdf",
        "document_id": "sop-storage-001",
        "title": "Sample Storage",
        "section_title": "Temperature Requirements",
        "page_start": 4,
        "page_end": 4,
    }

    first_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    first_store.add_documents(
        [Document(page_content="SOP-TEST: store at 2-8 C.", metadata=metadata)]
    )
    del first_store

    reopened_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    results = reopened_store.similarity_search("storage temperature", k=1)

    assert len(results) == 1
    assert results[0].page_content == "SOP-TEST: store at 2-8 C."
    assert results[0].metadata == metadata
