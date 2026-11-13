"""Offline, paired source-grounding audit for a four-way RAG report.

No GPT, PyLate, Gemini, network access, or tokens.
This intentionally separates graded answer correctness from source
provenance. An answer can be factually plausible yet cite the wrong paper.
Confidence intervals are exploratory paired question-bootstrap intervals,
not independent ground-truth scientific validation.

python -m rag_lab.evaluation.paired_audit \
  --summary eval/robustness_four_way_v5.json \
  --output eval/robustness_paired_audit.json
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
from statistics import mean,median

PAIRS=(("dense:naive","dense:crag"),("colbert:naive","colbert:crag"),
       ("dense:naive","colbert:naive"))
ARMS=("dense:naive","colbert:naive","dense:crag","colbert:crag")


def _score(row):
    return ((row.get("judge") or {}).get("answer_correctness") or 0.0)


def _interval(values, *, iterations=10000, seed=20261024):
    """Deterministic paired bootstrap: resample question-level differences."""
    n=len(values)
    if not n: return None
    rng=random.Random(seed)
    sampled=sorted(mean(values[rng.randrange(n)] for _ in range(n))
                   for _ in range(iterations))
    return [round(sampled[int(.025*iterations)],4),
            round(sampled[int(.975*iterations)-1],4)]


def audit(summary, question_types=None, *, min_score=0.90,
          iterations=10000,seed=20261024):
    qrows=summary.get("per_question") or {}
    if not qrows:raise ValueError("No per_question data")
    if summary.get("incomplete_question_ids"):
        raise ValueError("Incomplete four-arm report")
    expected=set(ARMS)
    if any(set(group)!=expected for group in qrows.values()):
        raise ValueError("Missing or unrecognized four-way arm")
    answers={}
    high_scoring_wrong_sources={}
    for arm in ARMS:
        values=[row[arm] for row in qrows.values()]
        answered=sum(bool(r["answered"]) for r in values)
        judged=sum(bool(r["judge"]) for r in values)
        if answered!=judged:
            raise ValueError(f"{arm} has unjudged answered records")
        full=sum(bool(r["answered"]) and r.get("paper_recall_at_4")==1
                 for r in values)
        scores=[_score(r) for r in values]
        times=[r["latency_ms"] for r in values if r.get("latency_ms") is not None]
        wrong=[{"id":qid,"judge_correctness":_score(rows[arm]),
                "paper_recall_at_4":rows[arm].get("paper_recall_at_4")}
               for qid,rows in qrows.items()
               if rows[arm]["answered"] and _score(rows[arm])>=min_score
               and rows[arm].get("paper_recall_at_4")!=1]
        high_scoring_wrong_sources[arm]=wrong
        answers[arm]={
            "n":len(values),"answered":answered,"judge_coverage":judged,
            "correctness_answered_only":round(sum(scores)/answered,4) if answered else None,
            "correctness_zero_abstentions":round(mean(scores),4),
            "fully_grounded_answer_rate":round(full/len(values),4),
            "correctness_and_complete_source_proxy":round(
                mean(_score(r) if r["answered"] and r.get("paper_recall_at_4")==1
                     else 0 for r in values),4),
            "mean_latency_seconds":round(mean(times)/1000,3) if times else None,
            "median_latency_seconds":round(median(times)/1000,3) if times else None,
            "wrong_source_high_correctness_count":len(wrong),
        }
    differences={}
    for left,right in PAIRS:
        delta=[_score(rows[right])-_score(rows[left]) for rows in qrows.values()]
        differences[f"{right} minus {left}"]={
            "mean":round(mean(delta),4),
            "paired_bootstrap_95_interval":_interval(
                delta,iterations=iterations,seed=seed),
            "wins":sum(x>0.01 for x in delta),
            "losses":sum(x<-.01 for x in delta),
            "ties_within_0_01":sum(abs(x)<=.01 for x in delta),
        }
    group_report={}
    if question_types is not None:
        if set(question_types)!=set(qrows):
            raise ValueError("Question-type IDs must match per_question IDs")
        for question_type in sorted(set(question_types.values())):
            ids=[qid for qid in qrows if question_types[qid]==question_type]
            group_report[question_type]={
                arm:{"n":len(ids),
                     "adjusted_correctness":round(
                         mean(_score(qrows[qid][arm]) for qid in ids),4),
                     "complete_source_coverage":sum(
                         bool(qrows[qid][arm]["answered"]) and
                         qrows[qid][arm].get("paper_recall_at_4")==1
                         for qid in ids)}
                for arm in ARMS}
    return {
        "purpose":"exploratory_paired_audit_not_independent_validation",
        "questions":len(qrows),"arms":answers,
        "paired_differences":differences,
        "wrong_source_high_correctness_cases":high_scoring_wrong_sources,
        "subgroups":group_report,
        "interpretation":{
            "source_coverage_gate":"Coverage proxy does not independently verify individual factual claims or quotations.",
            "zero_abstentions":"Zero treats abstention as zero task utility, not a fabricated scientific claim.",
            "bootstrap":"Exploratory bootstrap; repeated evaluation and model-authored references limit independent inference.",
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--eval-set",type=Path,
                   help="Optional dataset JSON to compute single/cross document subgroups")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    summary=json.loads(a.summary.read_text(encoding="utf-8"))
    kinds=None
    if a.eval_set:
        data=json.loads(a.eval_set.read_text(encoding="utf-8"))
        kinds={q["id"]:q["question_type"] for q in data["questions"]}
    result=audit(summary,kinds)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({
        "questions":result["questions"],
        "scores":result["arms"],
        "paired_differences":result["paired_differences"],
    },indent=2))


if __name__=="__main__":main()
