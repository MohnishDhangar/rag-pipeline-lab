"""Offline export tests: no Gemini keys, network access, or real index needed."""
import json
import pytest
from rag_lab.retrieval.export_colbert_leaves import (
    export, metadata_for_existing_texts, read_local_chroma
)


class Collection:
    def __init__(self,documents,metadatas):
        self.documents=documents
        self.metadatas=metadatas
    def count(self):
        return len(self.documents)
    def get(self,include=None):
        assert include==["documents","metadatas"]
        return {"documents":self.documents,"metadatas":self.metadatas}


class Client:
    def __init__(self,documents,metadatas):
        self.coll=Collection(documents,metadatas)
    def get_collection(self,name):
        assert name=="langchain"
        return self.coll


def test_recovers_old_colbert_mapping_independent_of_chroma_order(tmp_path,monkeypatch):
    (tmp_path/"data").mkdir()
    indexed={"chunk_0":"Second evidence","chunk_1":"First evidence"}
    (tmp_path/"data"/"id_to_text.json").write_text(json.dumps(indexed))
    (tmp_path/"data"/"colbert_index").mkdir()
    (tmp_path/"data"/"colbert_index"/"index.dat").write_text("fake")
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    result=export(tmp_path,client=Client(
        ["First evidence","Second evidence"],
        [{"source":"first.pdf","page":0},{"source":"second.pdf","page":3}],
    ))
    assert result["mode"]=="recovered_existing_index_metadata"
    mapping=json.loads((tmp_path/"data"/"id_to_metadata.json").read_text())
    assert mapping["chunk_0"]["source"]=="second.pdf"
    assert mapping["chunk_0"]["page"]==3
    assert mapping["chunk_1"]["source"]=="first.pdf"
    assert json.loads((tmp_path/"data"/"id_to_text.json").read_text())==indexed
    assert result["index_preserved"]
    # Rerun is idempotent and does not rewrite the existing mapping.
    assert export(tmp_path,client=Client(
        ["Second evidence","First evidence"],
        [{"source":"second.pdf","page":3},{"source":"first.pdf","page":0}],
    ))["chunks"]==2


def test_new_index_export_is_aligned(tmp_path):
    assert export(tmp_path,client=Client(
        ["Text A","Text B"],
        [{"source":"x.pdf","page":0},{"source":"y.pdf","page":1}]
    ))["mode"]=="new_index_export"
    mapping=json.loads((tmp_path/"data"/"id_to_metadata.json").read_text())
    leaves=json.loads((tmp_path/"data"/"leaf_texts.json").read_text())
    assert leaves==["Text A","Text B"]
    assert mapping["chunk_1"]["source"]=="y.pdf"


def test_duplicate_text_with_different_papers_fails_closed():
    with pytest.raises(ValueError,match="Ambiguous duplicate"):
        metadata_for_existing_texts(
            {"chunk_0":"same","chunk_1":"same"},
            ["same","same"],
            [{"source":"a.pdf","page":0},{"source":"b.pdf","page":1}]
        )


def test_old_colbert_with_changed_corpus_not_overwritten(tmp_path):
    (tmp_path/"data").mkdir()
    (tmp_path/"data"/"id_to_text.json").write_text('{"chunk_0":"old text"}')
    with pytest.raises(ValueError,match="differs from Chroma corpus"):
        export(tmp_path,client=Client(
            ["different text"],[{"source":"x.pdf","page":0}]
        ))
    assert not (tmp_path/"data"/"id_to_metadata.json").exists()


def test_existing_index_missing_text_mapping_refused(tmp_path):
    (tmp_path/"data"/"colbert_index").mkdir(parents=True)
    (tmp_path/"data"/"colbert_index"/"index.dat").write_text("fake")
    with pytest.raises(FileNotFoundError,match="index exists"):
        export(tmp_path,client=Client(
            ["Text A"],[{"source":"x.pdf","page":0}]
        ))


def test_local_chroma_export_calls_no_embedding_function():
    documents,metadatas=read_local_chroma(
        "/fake/path","langchain",
        client=Client(["text"],[{"source":"x.pdf","page":0}])
    )
    assert documents==["text"] and metadatas[0]["page"]==0
