"""Check ColBERT index and ID/metadata integrity before spending GPT calls.

python -m rag_lab.retrieval.colbert_health --root .
python -m rag_lab.retrieval.colbert_health --root . --query "XPINN interface conditions"

Without --query, NO model or remote API is instantiated; source mapping
is validated against the canonical filename registry and Chroma chunk count.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from rag_lab.retrieval.colbert_backend import validate_mappings
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.colbert_manifest import validate_manifest


def check(root, *, query=None, k=4, expected_chunks=None, strict_manifest=False,
          paper_id=None):
    root=Path(root)
    data=root/"data"
    with (data/"id_to_text.json").open(encoding="utf8") as f:
        texts=json.load(f)
    with (data/"id_to_metadata.json").open(encoding="utf8") as f:
        metadata=json.load(f)
    validate_mappings(texts,metadata)
    registry=PaperRegistry.from_eval(json.loads((root/"eval"/"eval_set.json").read_text()))
    source_counts={}
    unresolved=[]
    for chunk_id,meta in metadata.items():
        paper=registry.resolve(meta.get("source"))
        if not paper:
            unresolved.append({"id":chunk_id,"source":meta.get("source")})
        else:
            source_counts[paper]=source_counts.get(paper,0)+1
    if expected_chunks is not None and len(texts)!=expected_chunks:
        raise ValueError(f"ColBERT mapping has {len(texts)} chunks; expected {expected_chunks}")
    folder=data/"colbert_index"
    if not folder.is_dir() or not any(folder.iterdir()):
        raise FileNotFoundError(f"No PLAID index directory or files at {folder}")
    if unresolved:
        raise ValueError(f"{len(unresolved)} ColBERT chunks have unknown paper provenance: {unresolved[:3]}")
    result={"mapping_chunks":len(texts),"papers":source_counts,"index_path":str(folder),
            "manifest":validate_manifest(root,texts,metadata,strict=strict_manifest)}
    if paper_id is not None and query is None:
        raise ValueError("--paper-id requires --query")
    if query is not None:
        from rag_lab.retrieval.colbert_backend import ColBERTVectorStoreShim
        store=ColBERTVectorStoreShim(str(folder),texts,metadata)
        if paper_id is not None:
            paper_ids=set(registry.by_filename.values())
            if paper_id not in paper_ids:
                raise ValueError(f"Unknown paper ID {paper_id!r}")
            paths=sorted({meta["source"] for meta in metadata.values()
                          if registry.resolve(meta.get("source"))==paper_id})
            docs=[]
            for source in paths:
                docs.extend(store.similarity_search(query,k=k,
                                                    filter={"source":source}))
            docs=sorted(docs,key=lambda d:d.metadata.get("score") or 0,
                        reverse=True)[:k]
            result["subset_paper_id"]=paper_id
            result["subset_source_count"]=len(paths)
        else:
            docs=store.similarity_search(query,k=k)
        result["live_query"]={"question":query,"returned":len(docs),
                              "sources":[{"id":d.metadata["id"],
                                          "paper_id":registry.resolve(d.metadata["source"]),
                                          "page":d.metadata["page"],
                                          "score":d.metadata["score"],
                                          "excerpt":d.page_content[:180]} for d in docs]}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--query")
    p.add_argument("--k",type=int,default=4)
    p.add_argument("--expected-chunks",type=int,default=576)
    p.add_argument("--strict-manifest",action="store_true")
    p.add_argument("--paper-id",help="Limit live query to this canonical paper ID")
    a=p.parse_args()
    print(json.dumps(check(a.root,query=a.query,k=a.k,
                           expected_chunks=a.expected_chunks,
                           strict_manifest=a.strict_manifest,
                           paper_id=a.paper_id),indent=2))

if __name__=="__main__":main()
