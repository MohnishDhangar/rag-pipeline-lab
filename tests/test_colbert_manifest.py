import pytest
from rag_lab.retrieval.colbert_manifest import fingerprint,write_manifest,validate_manifest


def test_manifest_detects_mutated_texts_and_missing_manifest(tmp_path):
    root=tmp_path
    (root/"data").mkdir()
    texts={"chunk_0":"equation evidence"}
    meta={"chunk_0":{"source":"x.pdf","page":0}}
    assert validate_manifest(root,texts,meta)["status"]=="unverified_no_manifest"
    with pytest.raises(ValueError,match="no manifest"):
        validate_manifest(root,texts,meta,strict=True)
    write_manifest(root,texts,meta,model_name="test")
    assert validate_manifest(root,texts,meta,strict=True)["status"]=="matched"
    with pytest.raises(ValueError,match="fingerprint mismatch"):
        validate_manifest(root,{"chunk_0":"tampered"},meta)
