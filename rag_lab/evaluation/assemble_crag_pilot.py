"""Reuse v3 naive baselines while evaluating only corrected v4 CRAG arms.

This is purely an OFFLINE packaging operation. It does not invoke GPT,
Gemini, PyLate, OAuth, or judge models. Baseline rows and judge scores are
copied verbatim and validated by their exact generation fingerprints.

python -m rag_lab.evaluation.assemble_crag_pilot \
 --baseline-generated eval/generated_crag_v3_pilot.jsonl \
 --baseline-judged eval/judged_crag_v3_pilot.jsonl \
 --candidate-generated eval/generated_crag_v4_pilot.jsonl \
 --candidate-judged eval/judged_crag_v4_pilot.jsonl \
 --generated eval/generated_crag_v4_full.jsonl \
 --judged eval/judged_crag_v4_full.jsonl \
 --ids b1_gns_03 b1_cross_01 b2_cross_003
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.evaluation.identity import RUBRIC_VERSION, row_fingerprint
from rag_lab.evaluation.jsonl_io import read_jsonl

BACKENDS=("dense","colbert")
STRATEGIES=("naive","crag")
CRAG_VERSION="closed-corpus-crag-v4"


def assemble(baseline_generated,baseline_judged,candidate_generated,
             candidate_judged,generated,judged,ids,crag_version=CRAG_VERSION):
    expected_ids=tuple(ids)
    if not expected_ids or len(expected_ids)!=len(set(expected_ids)):
        raise ValueError("Pass distinct, nonempty --ids")
    for target in (generated,judged):
        if Path(target).exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {target}")
    selected={}
    for strategy,path in (("naive",baseline_generated),("crag",candidate_generated)):
        for row in read_jsonl(path):
            if row.get("strategy")!=strategy or row.get("id") not in expected_ids:
                continue
            key=(row.get("id"),row.get("backend"),strategy)
            if key[1] not in BACKENDS:
                continue
            if key in selected:
                raise ValueError(f"Duplicate generated pilot row: {key}")
            if strategy=="crag":
                if ((row.get("trace") or {}).get("crag") or {}).get("version")!=crag_version:
                    raise ValueError(f"Expected {crag_version} corrective trace, got {key}")
            selected[key]=row
    expected={(qid,backend,strategy)
              for qid in expected_ids for backend in BACKENDS
              for strategy in STRATEGIES}
    if set(selected)!=expected:
        raise ValueError(f"Four-arm pilot incomplete: missing={sorted(expected-set(selected))}, "
                         f"extra={sorted(set(selected)-expected)}")
    for qid in expected_ids:
        rows=[selected[qid,b,s] for b in BACKENDS for s in STRATEGIES]
        if len({r.get("question") for r in rows})!=1:
            raise ValueError(f"Question text changed since baseline: {qid}")
        if len({r.get("model") for r in rows})!=1:
            raise ValueError(f"GPT model changed since baseline: {qid}; rerun naive arms")
        if len({json.dumps(r.get("source_documents"),sort_keys=True)
                for r in rows})!=1:
            raise ValueError(f"Evaluation annotations changed: {qid}")
    judges={}
    for strategy,path in (("naive",baseline_judged),("crag",candidate_judged)):
        for row in read_jsonl(path):
            if row.get("strategy")!=strategy or row.get("id") not in expected_ids:
                continue
            if row.get("backend") not in BACKENDS:
                continue
            if row.get("judge_version")!=RUBRIC_VERSION:
                continue
            key=(row.get("id"),row.get("backend"),strategy,row.get("input_fingerprint"))
            if key in judges:
                raise ValueError(f"Duplicate judged pilot row: {key}")
            judges[key]=row
    selected_judges=[]
    for qid in expected_ids:
        for backend in BACKENDS:
            for strategy in STRATEGIES:
                row=selected[qid,backend,strategy]
                answered=bool(row.get("contexts") and row.get("answer") and
                    row.get("answer")!="Insufficient retrieved evidence to answer this question.")
                key=(qid,backend,strategy,row_fingerprint(row))
                j=judges.get(key)
                if answered and j is None:
                    raise ValueError(f"Missing judged answer for {qid}/{backend}/{strategy}")
                if j is not None:
                    selected_judges.append(j)
    rows=[selected[qid,backend,strategy] for qid in expected_ids
          for backend in BACKENDS for strategy in STRATEGIES]
    Path(generated).parent.mkdir(parents=True,exist_ok=True)
    Path(judged).parent.mkdir(parents=True,exist_ok=True)
    with Path(generated).open("x",encoding="utf-8") as g:
        for row in rows:g.write(json.dumps(row,ensure_ascii=False)+"\n")
    with Path(judged).open("x",encoding="utf-8") as j:
        for row in selected_judges:j.write(json.dumps(row,ensure_ascii=False)+"\n")
    return {"generation_rows":len(rows),"judged_rows":len(selected_judges),
            "naive_baseline_reused":len(expected_ids)*len(BACKENDS),
            "candidate_crag_version":crag_version,
            "candidate_crag_records":len(expected_ids)*len(BACKENDS)}


def main():
    p=argparse.ArgumentParser()
    for name in ("baseline-generated","baseline-judged","candidate-generated",
                 "candidate-judged","generated","judged"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--ids",nargs="+",required=True)
    p.add_argument("--crag-version",default=CRAG_VERSION,
                   help="Exact candidate CRAG trace version; e.g. closed-corpus-crag-v5")
    a=p.parse_args()
    kw={key.replace("-","_"):value for key,value in vars(a).items()}
    print(json.dumps(assemble(**kw),indent=2))


if __name__=="__main__":
    main()
