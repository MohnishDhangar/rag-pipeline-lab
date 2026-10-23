"""No-GPT dense vs ColBERT late-interaction retrieval benchmark.

python -m rag_lab.evaluation.retrieval_matrix \
  --backends dense colbert --limit 10 --output eval/colbert_retrieval_pilot.jsonl

Measures 4-retrieved-chunk ranking, passage provenance, and per-query time.
ColBERT runs locally; Gemini is called only by the dense-query encoder.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from rag_lab.evaluation.generate import create_backend
from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.provenance import ProvenanceStore
from rag_lab.evaluation.benchmark import read_jsonl
from rag_lab.utils import append_jsonl

VERSION="backend-retrieval-v1"


def benchmark(root, output, *, backends=("dense","colbert"), k=4, limit=None,
              ids=None, chroma_path=None, chroma_collection=None):
    root=Path(root)
    dataset=json.loads((root/"eval"/"eval_set.json").read_text())
    questions=dataset["questions"]
    if ids is not None:
        wanted=set(ids)
        unknown=wanted-{q["id"] for q in questions}
        if unknown: raise ValueError(f"Unknown question IDs: {sorted(unknown)}")
        questions=[q for q in questions if q["id"] in wanted]
    if limit is not None:questions=questions[:limit]
    registry=PaperRegistry.from_eval(dataset)
    existing=read_jsonl(output)
    complete={(r.get("version"),r.get("backend"),r.get("id"),r.get("k"))
              for r in existing}
    n=0
    for backend in backends:
        if backend not in ("dense","colbert"):
            raise ValueError("Supported backends: dense, colbert")
        base=create_backend(backend,root,chroma_path,chroma_collection)
        store=ProvenanceStore(base,registry)
        retriever=store.as_retriever(search_kwargs={"k":k})
        for q in questions:
            key=(VERSION,backend,q["id"],k)
            if key in complete:continue
            start=time.perf_counter()
            docs=retriever.invoke(q["question"])
            duration=round((time.perf_counter()-start)*1000,2)
            sources=[{
                "id":d.metadata.get("id"),
                "paper_id":d.metadata.get("paper_id"),
                "source":d.metadata.get("source"),
                "pdf_page_number":d.metadata.get("pdf_page_number"),
                "score":d.metadata.get("score"),
            } for d in docs]
            metrics=score_retrieval([s.get("paper_id") for s in sources],
                                    q["source_documents"],k=k)
            append_jsonl(output,{
                "version":VERSION,"backend":backend,
                "mode":"similarity" if backend=="dense" else "maxsim",
                "id":q["id"],"question_type":q.get("question_type"),
                "k":k,"latency_ms":duration,"sources":sources,**metrics,
            })
            complete.add(key);n+=1
    return n


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--output",type=Path,default=Path("eval/retrieval_backends_v1.jsonl"))
    p.add_argument("--backends",nargs="+",choices=["dense","colbert"],
                   default=["dense","colbert"])
    p.add_argument("--k",type=int,default=4)
    p.add_argument("--limit",type=int)
    p.add_argument("--ids",nargs="+")
    p.add_argument("--chroma-path",type=Path)
    p.add_argument("--chroma-collection")
    a=p.parse_args()
    if a.k<1:p.error("k must be positive")
    print("New backend retrieval experiments:",
          benchmark(a.root,a.output,backends=a.backends,k=a.k,limit=a.limit,
                    ids=a.ids,chroma_path=a.chroma_path,
                    chroma_collection=a.chroma_collection))


if __name__=="__main__":main()
