"""Compare paired GPT judge scores for baseline and candidate retrieval strategies.

No GPT or Gemini calls. Joins scores by an exact fingerprint of the generated
question, evidence and answer: stale judge rows cannot contaminate comparisons.

python -m rag_lab.evaluation.compare_answers \
  --generated eval/generated_gpt_adaptive_pilot.jsonl \
  --judged eval/gpt_judge_adaptive_pilot.jsonl \
  --output eval/adaptive_answer_comparison.json
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rag_lab.evaluation.benchmark import row_fingerprint
from rag_lab.evaluation.judge import RUBRIC_VERSION

METRICS=("faithfulness","answer_correctness","answer_relevance","citation_support")


def _read_strict(path):
    rows=[]
    for i,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as ex:
                raise ValueError(f"Malformed JSON line {i} in {path}") from ex
    return rows


def compare(generated, judged, output, baseline="naive", candidate="adaptive", backend="dense"):
    if baseline == candidate:
        raise ValueError("Comparison requires two different strategies")
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    scores = {}
    for score in _read_strict(judged):
        if score.get("judge_version") != RUBRIC_VERSION:
            continue
        key=(score.get("strategy"),score.get("backend"),score.get("id"),
             score.get("input_fingerprint"))
        if key in scores:
            # Duplicated completed scores are ambiguous, never pick silently.
            raise ValueError(f"Duplicate judge scores for {key}")
        scores[key]=score
    matched=defaultdict(dict)
    missing_judge=[]
    for row in _read_strict(generated):
        mode=row.get("strategy")
        if mode not in (baseline,candidate) or row.get("backend") != backend:
            continue
        key=(mode,backend,row["id"],row_fingerprint(row))
        judge=scores.get(key)
        if judge is None:
            missing_judge.append(f"{mode}/{row['id']}")
            continue
        if mode in matched[row["id"]]:
            raise ValueError(f"Duplicate generated strategy/question {mode}/{row['id']}")
        matched[row["id"]][mode]=(row,judge)
    pairs=[]
    for question,values in matched.items():
        if baseline not in values or candidate not in values:
            continue
        _, b=values[baseline]
        _, c=values[candidate]
        pairs.append({
            "id":question,
            "baseline":{m:b[m] for m in METRICS},
            "candidate":{m:c[m] for m in METRICS},
            "delta_candidate_minus_baseline":{m:c[m]-b[m] for m in METRICS},
        })
    result={
        "purpose":"paired_exploratory_GPT_judge_not_human_ground_truth",
        "judge_version":RUBRIC_VERSION,"backend":backend,
        "baseline":baseline,"candidate":candidate,
        "paired_n":len(pairs),"unmatched_judge_rows":missing_judge,
        "means":{
            m:{"baseline":round(mean(x["baseline"][m] for x in pairs),6),
               "candidate":round(mean(x["candidate"][m] for x in pairs),6),
               "delta":round(mean(x["delta_candidate_minus_baseline"][m] for x in pairs),6),
               "candidate_wins":sum(x["delta_candidate_minus_baseline"][m]>0 for x in pairs),
               "baseline_wins":sum(x["delta_candidate_minus_baseline"][m]<0 for x in pairs),
               "ties":sum(x["delta_candidate_minus_baseline"][m]==0 for x in pairs)}
            for m in METRICS
        } if pairs else {},
        "pairs":pairs,
    }
    with output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
        f.write("\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generated",type=Path,required=True)
    p.add_argument("--judged",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--baseline",default="naive")
    p.add_argument("--candidate",default="adaptive")
    p.add_argument("--backend",default="dense")
    a=p.parse_args()
    res=compare(a.generated,a.judged,a.output,a.baseline,a.candidate,a.backend)
    print(json.dumps({k:res[k] for k in ("paired_n","unmatched_judge_rows","means")},indent=2))


if __name__=="__main__":main()
