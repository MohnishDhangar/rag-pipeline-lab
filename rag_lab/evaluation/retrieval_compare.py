"""Zero-GPT retrieval ablations using Gemini embeddings and existing Chroma indexes.

python -m rag_lab.evaluation.retrieval_compare --modes similarity mmr \
  --k 4 --limit 10 --output eval/retrieval_ablation_v2.jsonl

Retrieval algorithms never see gold 'source_documents'; only the evaluator does.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from rag_lab.retrieval.backend_factory import create_backend
from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.evaluation.jsonl_io import read_jsonl
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.provenance import ProvenanceStore
from rag_lab.utils import append_jsonl

VERSION = "retrieval-v2"


def compare(root, output, modes=("similarity","mmr"), k=4, limit=None,
            chroma_path=None, chroma_collection=None, fetch_k=20, lambda_mult=.5):
    root=Path(root)
    dataset=json.loads((root/"eval"/"eval_set.json").read_text())
    questions=dataset["questions"]
    if limit is not None: questions=questions[:limit]
    registry=PaperRegistry.from_eval(dataset)
    # No GPT transport constructed: this is a retrieval-only experiment.
    base=create_backend("dense",root,chroma_path,chroma_collection)
    store=ProvenanceStore(base,registry)
    completed={(r.get("version"),r.get("id"),r.get("mode"),r.get("k"),
                r.get("fetch_k"),r.get("lambda_mult"))
               for r in read_jsonl(output)}
    count=0
    for mode in modes:
        if mode not in ("similarity","mmr"): raise ValueError("Supported modes: similarity, mmr")
        search_kwargs={"k":k}
        if mode=="mmr":
            search_kwargs.update(fetch_k=fetch_k,lambda_mult=lambda_mult)
        retriever=base.as_retriever(search_type=mode,search_kwargs=search_kwargs)
        for question in questions:
            key=(VERSION,question["id"],mode,k,fetch_k if mode=="mmr" else None,
                 lambda_mult if mode=="mmr" else None)
            if key in completed:continue
            start=time.perf_counter()
            docs=store._decorate(retriever.invoke(question["question"]))
            latency=round((time.perf_counter()-start)*1000,2)
            src=[d.metadata for d in docs]
            metrics=score_retrieval([s.get("paper_id") for s in src],
                                    question.get("source_documents"),k=k)
            result={"version":VERSION,"id":question["id"],"mode":mode,"k":k,
                    "fetch_k":fetch_k if mode=="mmr" else None,
                    "lambda_mult":lambda_mult if mode=="mmr" else None,
                    "backend":"dense","latency_ms":latency,
                    "question_type":question.get("question_type"),
                    "sources":[{"id":s.get("id"),"paper_id":s.get("paper_id"),
                                "pdf_page_number":s.get("pdf_page_number"),
                                "source":s.get("source")} for s in src],
                    **metrics}
            append_jsonl(output,result)
            count+=1
    return count


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--output",type=Path,default=Path("eval/retrieval_ablation_v2.jsonl"))
    p.add_argument("--modes",nargs="+",choices=["similarity","mmr"],
                   default=["similarity","mmr"])
    p.add_argument("--k",type=int,default=4)
    p.add_argument("--fetch-k",type=int,default=20)
    p.add_argument("--lambda-mult",type=float,default=.5)
    p.add_argument("--limit",type=int)
    p.add_argument("--chroma-path",type=Path)
    p.add_argument("--chroma-collection")
    a=p.parse_args()
    if not (0<=a.lambda_mult<=1):p.error("lambda-mult must be in [0,1]")
    print("New retrieval experiments:",
          compare(a.root,a.output,a.modes,a.k,a.limit,a.chroma_path,
                  a.chroma_collection,a.fetch_k,a.lambda_mult))


if __name__=="__main__":main()
