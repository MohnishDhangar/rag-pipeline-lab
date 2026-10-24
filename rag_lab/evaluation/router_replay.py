"""Replay text-only adaptive routing on saved similarity/MMR results.

No model or index calls. This is an *exploratory* analysis because the 63
questions were already inspected. Do not claim a held-out performance win.

python -m rag_lab.evaluation.router_replay \
  --input eval/retrieval_ablation_63.jsonl \
  --output eval/router_replay_63.json
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.query_router import classify_question, VERSION

METRICS = ("chunk_precision_at_k", "paper_recall_at_k", "mrr_at_k", "ndcg_at_k")
MODES = ("similarity", "mmr", "adaptive")
REPORT_VERSION = "router-replay-v1"


def _summary(items):
    return {
        "n": len(items),
        **{metric: round(mean(item[metric] for item in items), 6)
           if items and all(item[metric] is not None for item in items)
           else None for metric in METRICS},
        "full_paper_recall_count": sum(item["paper_recall_at_k"] == 1 for item in items),
        "mean_latency_ms": round(mean(item["latency_ms"] for item in items), 3)
        if items else None,
    }


def replay(input_path, output_path, eval_path):
    inp, out = Path(input_path).resolve(), Path(output_path).resolve()
    if inp == out:
        raise ValueError("Do not overwrite retrieved evidence")
    if out.exists():
        raise FileExistsError(str(out))
    dataset = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in dataset["questions"]}
    registry = PaperRegistry.from_eval(dataset)

    pairs = defaultdict(dict)
    with inp.open(encoding="utf-8") as f:
        for n, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)  # strict parsing (no silent omission)
            id_ = row["id"]
            mode = row["mode"]
            if id_ not in questions or mode not in ("similarity", "mmr"):
                raise ValueError(f"Unexpected question/mode at line {n}")
            if mode in pairs[id_]:
                raise ValueError(f"Duplicate {id_}:{mode}")
            paper_ids=[]
            for src in row.get("sources", []):
                claimed = src.get("paper_id")
                mapped = registry.resolve(src.get("source"))
                if claimed and mapped and claimed != mapped:
                    raise ValueError(f"Conflicting source label: {id_}:{claimed}/{mapped}")
                paper_ids.append(claimed or mapped)
            scores=score_retrieval(paper_ids,questions[id_].get("source_documents"),
                                  k=int(row["k"]))
            pairs[id_][mode] = {**scores, "latency_ms":row["latency_ms"],
                                "paper_ids":paper_ids}
    if set(pairs) != set(questions):
        missing=set(questions) - set(pairs)
        extra=set(pairs) - set(questions)
        raise ValueError(f"Expected all questions: missing={sorted(missing)} extra={sorted(extra)}")
    for id_, value in pairs.items():
        if set(value) != {"similarity", "mmr"}:
            raise ValueError(f"Missing paired run for {id_}")
    rows=[]
    confusion=Counter()
    for id_, q in questions.items():
        decision=classify_question(q["question"])  # ONLY text is provided
        selected=pairs[id_][decision.search_type]
        truth=q.get("question_type")
        confusion[(truth,decision.question_type)]+=1
        rows.append({
            "id":id_, "gold_question_type_for_analysis_only":truth,
            "predicted_question_type":decision.question_type,
            "router_rule":decision.rule, "chosen_mode":decision.search_type,
            "scores":{mode:{k:v for k,v in pairs[id_][mode].items()
                             if k != "paper_ids"} for mode in ("similarity","mmr")},
            "selected":{k:v for k,v in selected.items() if k != "paper_ids"},
            "similarity_papers":pairs[id_]["similarity"]["paper_ids"],
            "mmr_papers":pairs[id_]["mmr"]["paper_ids"],
        })
    by_method={mode:[] for mode in MODES}
    by_type=defaultdict(lambda:{mode:[] for mode in MODES})
    for r in rows:
        record={**r["selected"]}
        for mode in ("similarity","mmr"):
            by_method[mode].append(r["scores"][mode])
            by_type[r["gold_question_type_for_analysis_only"]][mode].append(r["scores"][mode])
        by_method["adaptive"].append(record)
        by_type[r["gold_question_type_for_analysis_only"]]["adaptive"].append(record)
    report={
        "report_version":REPORT_VERSION,
        "router_version":VERSION,
        "experiment_type":"exploratory_offline_replay_not_held_out",
        "gold_labels_visible_only_to_evaluator":True,
        "question_count":len(rows),
        "confusion":{f"{gold} -> {pred}":count for (gold,pred),count in sorted(confusion.items())},
        "router_accuracy_on_existing_set":round(sum(g==p for (g,p),n in confusion.items() for _ in range(n))/len(rows),6),
        "overall":{mode:_summary(by_method[mode]) for mode in MODES},
        "by_question_type":{group:{mode:_summary(items) for mode,items in modes.items()}
                            for group,modes in by_type.items()},
        "paired_questions":rows,
    }
    # Exclusive creation: protects earlier experiments.
    with out.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
        f.write("\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--eval-set",type=Path,default=Path("eval/eval_set.json"))
    args=p.parse_args()
    result=replay(args.input,args.output,args.eval_set)
    print(json.dumps({key:result[key] for key in
                      ("report_version","question_count","confusion","overall","by_question_type")},
                     indent=2))


if __name__=="__main__":main()
