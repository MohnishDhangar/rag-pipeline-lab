"""Four-way dense/ColBERT × naive/CRAG result summary with judge linkage.

python -m rag_lab.evaluation.four_way_report \
  --generated eval/generated_four_way.jsonl \
  --judged eval/judged_four_way.jsonl \
  --output eval/four_way_summary.json

No GPT or Gemini calls; CRAG abstentions are included in denominators.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from rag_lab.evaluation.jsonl_io import read_jsonl
from rag_lab.evaluation.identity import row_fingerprint, RUBRIC_VERSION
from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.knowledge.paper_registry import PaperRegistry

ARMS=(("dense","naive"),("colbert","naive"),("dense","crag"),("colbert","crag"))
JUDGE_KEYS=("faithfulness","answer_correctness","answer_relevance","citation_support")


def report(generated, judged, eval_set, output):
    output=Path(output)
    if output.exists():raise FileExistsError(str(output))
    data=json.loads(Path(eval_set).read_text(encoding="utf-8"))
    truth={q["id"]:q["source_documents"] for q in data["questions"]}
    registry=PaperRegistry.from_eval(data)
    judge_map={}
    for j in read_jsonl(judged):
        if j.get("judge_version")!=RUBRIC_VERSION:continue
        key=(j.get("id"),j.get("backend"),j.get("strategy"),j.get("input_fingerprint"))
        if key in judge_map:raise ValueError(f"Duplicate judge score: {key}")
        judge_map[key]=j
    arm_rows=defaultdict(list)
    by_question=defaultdict(dict)
    for row in read_jsonl(generated):
        key=(row.get("backend"),row.get("strategy"))
        if key not in ARMS:continue
        qid=row["id"]
        if qid not in truth:raise ValueError(f"Unknown question: {qid}")
        if key in by_question[qid]:
            raise ValueError(f"Duplicate generation for {qid}/{key}")
        sources=row.get("sources") or []
        ids=[(s.get("paper_id") or registry.resolve(s.get("source")))
             for s in sources]
        metrics=score_retrieval(ids,truth[qid],k=4)
        has_answer=bool(row.get("contexts") and row.get("answer") and
                        row["answer"]!="Insufficient retrieved evidence to answer this question.")
        j=judge_map.get((qid,*key,row_fingerprint(row))) if has_answer else None
        summary={"id":qid,"answered":has_answer,"paper_recall_at_4":metrics["paper_recall_at_k"],
                 "chunk_precision_at_4":metrics["chunk_precision_at_k"],
                 "latency_ms":row.get("latency_ms"),
                 "gpt_calls":(row.get("trace") or {}).get("gpt_requests",0),
                 "grade_calls":((row.get("trace") or {}).get("crag") or {}).get("grader_calls",0),
                 "judge":{m:j[m] for m in JUDGE_KEYS} if j else None}
        by_question[qid][f"{key[0]}:{key[1]}"]=summary
        arm_rows[key].append(summary)
    arms={}
    for key in ARMS:
        rows=arm_rows[key]
        scorable=[r for r in rows if r["paper_recall_at_4"] is not None]
        judge_rows=[r for r in rows if r["judge"]]
        latency_values=[r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        arms[f"{key[0]}:{key[1]}"]={
            "n":len(rows),"answered":sum(x["answered"] for x in rows),
            "abstained_or_missing":sum(not x["answered"] for x in rows),
            "paper_recall_at_4":mean(r["paper_recall_at_4"] for r in scorable) if scorable else None,
            "chunk_precision_at_4":mean(r["chunk_precision_at_4"] for r in scorable) if scorable else None,
            "avg_latency_ms":mean(latency_values) if latency_values else None,
            "avg_gpt_calls":mean(r["gpt_calls"] for r in rows) if rows else None,
            "judge_coverage":len(judge_rows),
            **{"judge_"+m:mean(r["judge"][m] for r in judge_rows)
               if judge_rows else None for m in JUDGE_KEYS},
        }
    complete_ids=[qid for qid,arms_for_q in by_question.items()
                  if len(arms_for_q)==len(ARMS)]
    result={"purpose":"exploratory_4way_no_independent_human_validation",
            "arms":arms,"per_question":by_question,
            "all_four_arms_count":len(complete_ids),
            "total_questions_observed":len(by_question),
            "incomplete_question_ids":[qid for qid in sorted(by_question)
                                       if qid not in complete_ids],
            "all_four_arms_judged_count":sum(
                len(arms_for_q)==len(ARMS) and all(
                    row["judge"] is not None or not row["answered"]
                    for row in arms_for_q.values())
                for arms_for_q in by_question.values()),
            "limitations":"Scores are GPT-judged on answered rows; abstentions must remain in denominator."}
    with output.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write("\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generated",required=True)
    p.add_argument("--judged",required=True)
    p.add_argument("--eval-set",default="eval/eval_set.json")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=report(a.generated,a.judged,a.eval_set,a.output)
    print(json.dumps({"arms":result["arms"],"all_four_arms_count":
                      result["all_four_arms_count"]},indent=2))


if __name__=="__main__":main()
