"""Export aligned Chroma passage text and PDF provenance WITHOUT embeddings/API keys.

This reads local Chroma directly through chromadb.PersistentClient. It never
generates embeddings or touches the PLAID index.

When a legacy ColBERT id_to_text.json exists, join current Chroma metadata
to those exact stored texts. Do NOT presume Chroma's iteration order matches
a previously indexed ColBERT corpus, and never change existing mappings
without an explicit rebuild. Identical text with differing PDF metadata is
ambiguous: fail closed rather than invent source/page identity.

python -m rag_lab.retrieval.export_colbert_leaves --root .
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def metadata_for_existing_texts(indexed_texts, documents, metadatas):
    """Match legacy ColBERT IDs to Chroma metadata via exact passage text.

    Reject cases where multiple PDF occurrences of the same text disagree in
    their metadata: text-only ColBERT mappings cannot distinguish those chunks.
    """
    if len(documents) != len(metadatas):
        raise ValueError("Local Chroma returned different document/metadata counts")
    by_text = defaultdict(list)
    for text, meta in zip(documents, metadatas):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Chroma has an empty/non-string passage")
        if not isinstance(meta, dict) or not meta.get("source") or meta.get("page") is None:
            raise ValueError("Chroma chunk has missing source/page provenance")
        by_text[text].append(meta)
    indexed_counts = defaultdict(int)
    for key, text in indexed_texts.items():
        if not isinstance(key, str) or not isinstance(text, str):
            raise ValueError("Existing ColBERT id_to_text must map string IDs to strings")
        indexed_counts[text] += 1
    current_counts = {text: len(group) for text, group in by_text.items()}
    if dict(indexed_counts) != current_counts:
        raise ValueError(
            "Existing ColBERT text mapping differs from Chroma corpus: "
            "wrong/missing chunks or changed passages. Do not overwrite the "
            "old index; rebuild a new aligned index after backing it up."
        )
    result={}
    for key, text in indexed_texts.items():
        matches=by_text[text]
        # Ignore harmless differences in key ordering, but never silently
        # choose between different source/page occurrences of identical text.
        unique={json.dumps(meta,sort_keys=True) for meta in matches}
        if len(unique)>1:
            raise ValueError(
                f"Ambiguous duplicate ColBERT passage {key!r}: identical text "
                "has multiple distinct Chroma metadata records. Cannot "
                "reconstruct source identity from id_to_text alone."
            )
        result[key]=dict(matches[0])
    return result


def read_local_chroma(chroma_path, collection, *, client=None):
    location=Path(chroma_path).expanduser().resolve()
    if client is None:
        if not (location / "chroma.sqlite3").is_file():
            raise FileNotFoundError(
                f"No existing Chroma database at {location}; "
                "supply --chroma-path pointing to your populated index"
            )
        import chromadb
        client=chromadb.PersistentClient(path=str(location))
    try:
        selected=client.get_collection(name=collection)
    except Exception as exc:
        raise ValueError(
            f"Chroma collection {collection!r} not found at {location}"
        ) from exc
    if selected.count()<1:
        raise ValueError(f"Chroma collection {collection!r} is empty")
    # Get with BOTH fields in one operation so index-by-position refers to the
    # same Chroma chunk; no embedding_function or network requests involved.
    raw=selected.get(include=["documents","metadatas"])
    documents=raw.get("documents") or []
    metadatas=raw.get("metadatas") or []
    if len(documents) != selected.count() or len(documents)!=len(metadatas):
        raise ValueError("Incomplete or misaligned Chroma export")
    return documents,metadatas


def export(root, chroma_path=None, chroma_collection=None, *, client=None):
    root=Path(root)
    data=root/"data"
    location=chroma_path or os.environ.get("RAG_CHROMA_PATH") or (
        data/"chroma_naive_gemini"
    )
    collection=chroma_collection or os.environ.get("RAG_CHROMA_COLLECTION","langchain")
    documents,metadatas=read_local_chroma(location,collection,client=client)
    data.mkdir(parents=True,exist_ok=True)
    text_path=data/"id_to_text.json"
    metadata_path=data/"id_to_metadata.json"
    leaf_path=data/"leaf_texts.json"
    has_index=(data/"colbert_index").is_dir() and any(
        (data/"colbert_index").iterdir()
    )

    if text_path.exists():
        texts=json.loads(text_path.read_text(encoding="utf-8"))
        if not isinstance(texts,dict) or not texts:
            raise ValueError("Existing ColBERT ID-to-text mapping is empty/invalid")
        mapping=metadata_for_existing_texts(texts,documents,metadatas)
        leaf_docs=[texts[k] for k in texts]
        mode="recovered_existing_index_metadata"
    else:
        if has_index:
            raise FileNotFoundError(
                "PLAID index exists but data/id_to_text.json is missing; "
                "cannot infer chunk-ID alignment. Keep the old index and "
                "restore its mapping from backup or rebuild separately."
            )
        if any(not isinstance(text,str) or not text.strip() for text in documents):
            raise ValueError("Cannot export blank/non-string Chroma passages")
        if any(not isinstance(m,dict) or not m.get("source") or m.get("page") is None
               for m in metadatas):
            raise ValueError("Cannot export Chroma passages missing PDF source/page")
        mapping={f"chunk_{i}":dict(meta) for i,meta in enumerate(metadatas)}
        leaf_docs=list(documents)
        mode="new_index_export"

    if metadata_path.exists():
        existing=json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing!=mapping:
            raise FileExistsError(
                "id_to_metadata.json already exists and differs from verified "
                "Chroma metadata. Back it up and inspect the differences; "
                "not overwriting an existing index mapping."
            )
    if leaf_path.exists():
        existing=json.loads(leaf_path.read_text(encoding="utf-8"))
        if existing!=leaf_docs:
            if has_index:
                # The existing ColBERT ID-to-text mapping is authoritative.
                # A stale leaf_texts.json is not used for retrieval and must
                # not be silently overwritten.
                raise ValueError(
                    "leaf_texts.json conflicts with the existing ColBERT "
                    "ID-to-text mapping. Inspect before rebuilding or replacing."
                )
            raise FileExistsError(
                "leaf_texts.json differs from the verified export; "
                "refusing to overwrite it"
            )

    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(mapping,ensure_ascii=False,indent=2),encoding="utf-8"
        )
    if not leaf_path.exists():
        leaf_path.write_text(
            json.dumps(leaf_docs,ensure_ascii=False),encoding="utf-8"
        )
    return {"chunks":len(documents),"mode":mode,"collection":collection,
            "used_api_key":False,"index_preserved":has_index,
            "metadata_path":str(metadata_path)}


def main():
    p=argparse.ArgumentParser(
        description="Export PDF passage metadata from local Chroma without API keys"
    )
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--chroma-path",type=Path,
                   help="Existing populated Chroma directory (defaults to data/chroma_naive_gemini)")
    p.add_argument("--chroma-collection",
                   help="Collection name (defaults to langchain)")
    a=p.parse_args()
    print(json.dumps(export(a.root,a.chroma_path,a.chroma_collection),indent=2))


if __name__=="__main__":
    main()
