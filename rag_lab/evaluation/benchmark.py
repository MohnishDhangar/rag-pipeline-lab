"""Resumable offline retrieval-only and GPT judge evaluation.

python -m rag_lab.evaluation.benchmark retrieval --input eval/generated_gpt.jsonl
python -m rag_lab.evaluation.benchmark judge --input eval/generated_gpt.jsonl
"""
import argparse, json
from pathlib import Path

from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.evaluation.judge import evaluate_row, RUBRIC_VERSION
from rag_lab.utils import append_jsonl
from rag_lab.llms.gpt_adapter import InferenceError

def read_jsonl(path):
    if not Path(path).exists(): return []
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
    return rows

def benchmark(input_path, output_path, mode, registry=None):
    completed = {
        (r.get("strategy"),r.get("backend"),r.get("id"),r.get("judge_version") if mode=="judge" else None)
        for r in read_jsonl(output_path)
    }
    successful = 0
    for row in read_jsonl(input_path):
        if not row.get("answer") or not row.get("contexts"): continue  # no GPT judging on missing evidence
        key=(row.get("strategy"),row.get("backend"),row.get("id"),
             RUBRIC_VERSION if mode=="judge" else None)
        if key in completed: continue
        if mode=="retrieval":
            sources=row.get("sources") or []
            paper_ids=[]
            for doc in sources:
                if not isinstance(doc,dict):
                    paper_ids.append(None)
                else:
                    paper_ids.append(doc.get("paper_id") or
                                     (registry.resolve(doc.get("source")) if registry else None))
            scores=score_retrieval(paper_ids,row.get("source_documents"),k=4)
        else:
            try: scores=evaluate_row(row)
            except (InferenceError, ConnectionError, TimeoutError) as exc:
                print("Judge stopped; checkpoint preserved:",type(exc).__name__,str(exc))
                break
        append_jsonl(output_path, {"strategy":row.get("strategy"),"backend":row.get("backend"),
                                    "id":row.get("id"),**scores})
        completed.add(key)
        successful+=1
    return successful

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["retrieval","judge"])
    parser.add_argument("--input",default="eval/generated_gpt.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--eval",default="eval/eval_set.json",help="Ground-truth registry (no retrieval leakage)")
    args=parser.parse_args()
    output=args.output or ("eval/retrieval_metrics.jsonl" if args.mode=="retrieval"
                           else "eval/gpt_judge_scores.jsonl")
    registry=PaperRegistry.from_eval(json.loads(Path(args.eval).read_text()))
    print("New rows:",benchmark(args.input,output,args.mode,registry))

if __name__=="__main__": main()
