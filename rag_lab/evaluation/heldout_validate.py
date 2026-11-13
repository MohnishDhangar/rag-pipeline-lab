"""Create and structurally validate a pre-registered scientific RAG holdout.

This tool never generates test answers or truth labels. A qualified human must
write and independently verify each reference against the source PDFs.
Structural disjointness does NOT prove statistical independence.

Example:
python -m rag_lab.evaluation.heldout_validate init \
  --development eval/eval_set.json --output eval/heldout_eval_set.json
python -m rag_lab.evaluation.heldout_validate check \
  --development eval/eval_set.json --candidate eval/heldout_eval_set.json \
  --minimum 12
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from rag_lab.retrieval.source_aware import requested_sources


def _normalize(question):
    return re.sub(r"\s+"," ",str(question).lower()).strip()


def make_template(development):
    dataset=json.loads(Path(development).read_text(encoding="utf-8"))
    return {
        "metadata":{
            "papers":dataset["metadata"]["papers"],
            "evaluation_protocol":"heldout-scientific-rag-v1",
            "human_reference_reviewed":False,
            "authoring_notes":"Write new questions without seeing model predictions; verify reference answers against source PDFs before the first run."
        },
        "questions":[],
    }


def check(development,candidate,minimum=12,near_duplicate_threshold=0.88):
    a=json.loads(Path(development).read_text(encoding="utf-8"))
    b=json.loads(Path(candidate).read_text(encoding="utf-8"))
    meta=b.get("metadata") or {}
    papers=meta.get("papers",[])
    if not isinstance(papers,list) or not papers:
        raise ValueError("Holdout metadata.papers must include the registry")
    originals={p["short_id"] for p in a["metadata"]["papers"]}
    current={p["short_id"] for p in papers}
    if current!=originals or len(papers)!=len(originals):
        raise ValueError("Holdout paper registry differs from development corpus")
    if meta.get("evaluation_protocol")!="heldout-scientific-rag-v1":
        raise ValueError("Set metadata.evaluation_protocol to heldout-scientific-rag-v1")
    questions=b.get("questions")
    if not isinstance(questions,list) or len(questions)<minimum:
        raise ValueError(f"Need at least {minimum} independent held-out questions")
    original_ids={q["id"] for q in a["questions"]}
    original_norms=[_normalize(q["question"]) for q in a["questions"]]
    seen_ids=set()
    seen_questions=set()
    from rag_lab.knowledge.paper_registry import PaperRegistry
    registry=PaperRegistry.from_eval(b)
    for q in questions:
        id=q.get("id")
        if not isinstance(id,str) or not id.strip() or id in seen_ids or id in original_ids:
            raise ValueError(f"Invalid, repeated or development-set ID: {id!r}")
        seen_ids.add(id)
        question=q.get("question")
        if not isinstance(question,str) or len(question.strip())<25:
            raise ValueError(f"Missing/short question for {id}")
        normalized=_normalize(question)
        if normalized in seen_questions:
            raise ValueError(f"Repeated question wording: {id}")
        seen_questions.add(normalized)
        if max(SequenceMatcher(None,normalized,ref).ratio()
               for ref in original_norms) >= near_duplicate_threshold:
            raise ValueError(f"{id} looks too similar to a development question")
        if requested_sources(question,registry).unresolved_scope:
            raise ValueError(f"{id} relies on unspecified paper/batch context")
        sources=q.get("source_documents")
        if not isinstance(sources,list) or not sources or any(
            not isinstance(src,str) or src not in current for src in sources
        ) or len(sources)!=len(set(sources)):
            raise ValueError(f"Missing/invalid gold source labels for {id}")
        expected_type="cross_document" if len(sources)>1 else "single_document"
        if q.get("question_type")!=expected_type:
            raise ValueError(f"{id}: question_type must be {expected_type}")
        ref=q.get("ground_truth")
        if not isinstance(ref,str) or len(ref.strip())<30:
            raise ValueError(f"{id}: reference needs a substantive human-verified explanation")
    canonical=json.dumps(b,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return {
        "count":len(questions),
        "unique_ids":len(seen_ids),
        "source_registry_count":len(current),
        "single_document":sum(q["question_type"]=="single_document" for q in questions),
        "cross_document":sum(q["question_type"]=="cross_document" for q in questions),
        "canonical_sha256":hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "review_declaration":bool(meta.get("human_reference_reviewed",False)),
        "status":"structurally_disjoint_unreviewed" if not meta.get("human_reference_reviewed") else
                 "structurally_disjoint_review_self_declared",
        "caveat":"This automated check does NOT prove independent authorship, actual source correctness, or lack of semantic overlap. Verify and freeze the questions BEFORE looking at new model outputs."
    }


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode",required=True)
    for name in ("init","check"):
        x=sub.add_parser(name)
        x.add_argument("--development",type=Path,default=Path("eval/eval_set.json"))
        if name=="init":
            x.add_argument("--output",type=Path,required=True)
        else:
            x.add_argument("--candidate",type=Path,required=True)
            x.add_argument("--minimum",type=int,default=12)
    args=p.parse_args()
    if args.mode=="init":
        dest=args.output
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("x",encoding="utf-8") as handle:
            json.dump(make_template(args.development),handle,indent=2,ensure_ascii=False)
            handle.write("\n")
        print(f"Created empty human-authoring template at {dest}")
    else:
        print(json.dumps(check(args.development,args.candidate,
                               minimum=args.minimum),indent=2))


if __name__=="__main__":main()
