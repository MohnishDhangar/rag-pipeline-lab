"""Hash the exact ID/text/metadata mapping embedded by the ColBERT indexer.

A valid manifest is written ONLY after PyLate successfully indexes documents.
Existing indexes without manifests remain usable but are NOT independently
verified as having been built from the current texts. Rebuild intentionally.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path


def fingerprint(texts,metadata):
    pairs=[{"id":key,"text":texts[key],"metadata":metadata[key]}
           for key in sorted(texts)]
    payload=json.dumps(pairs,sort_keys=True,ensure_ascii=False,separators=(",",":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_path(root):
    return Path(root)/"data"/"colbert_manifest.json"


def write_manifest(root,texts,metadata,*,model_name):
    record={"schema":"colbert-index-manifest-v1",
            "model":model_name,"chunks":len(texts),
            "mapping_sha256":fingerprint(texts,metadata)}
    manifest_path(root).write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    return record


def validate_manifest(root,texts,metadata,*,strict=False):
    path=manifest_path(root)
    if not path.exists():
        if strict:
            raise ValueError("ColBERT index has no manifest; index/text alignment unverified")
        return {"status":"unverified_no_manifest"}
    record=json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema")!="colbert-index-manifest-v1":
        raise ValueError("Unexpected ColBERT manifest schema")
    if record["chunks"]!=len(texts) or record["mapping_sha256"]!=fingerprint(texts,metadata):
        raise ValueError("ColBERT indexed text/metadata fingerprint mismatch; rebuild index")
    return {"status":"matched","model":record["model"],
            "mapping_sha256":record["mapping_sha256"]}
