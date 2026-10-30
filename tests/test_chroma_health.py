import pytest
from rag_lab.retrieval.chroma_health import inspect_index, require_populated_index


def test_missing_index_does_not_create_empty_index(tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        inspect_index(path)
    assert not path.exists()


def test_existing_index_must_have_documents(tmp_path):
    import chromadb
    path = tmp_path / "store"
    collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection("langchain")
    with pytest.raises(RuntimeError,match="missing or empty"):
        require_populated_index(path, "langchain")
    collection.add(ids=["doc1"],embeddings=[[1.0,0.0]],
                   documents=["Physics informed neural networks"])
    assert require_populated_index(path, "langchain") == 1
