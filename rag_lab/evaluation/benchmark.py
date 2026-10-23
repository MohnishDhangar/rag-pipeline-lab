"""Resumable offline retrieval-only and GPT judge evaluation.

python -m rag_lab.evaluation.benchmark retrieval --input eval/generated_gpt.jsonl
python -m rag_lab.evaluation.benchmark judge --input eval/generated_gpt.jsonl
"""
import argparse, json
from pathlib import Path

from rag_lab.evaluation.metrics import score_retrieval
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

def benchmark(input_path, output_path, mode):
    completed = {
        (r.get("strategy"),r.get("backend"),r.get("id"),r.get("judge_version") if mode=="judge" else None)
        for r in read_jsonl(output_path)
    }
    successful = 0
    for row in read_jsonl(input_path):
        if not row.get("answer"): continue
        key=(row.get("strategy"),row.get("backend"),row.get("id"),
             RUBRIC_VERSION if mode=="judge" else None)
        if key in completed: continue
        if mode=="retrieval":
            sources=row.get("sources") or []
            paper_ids=[d.get("paper_id") for d in sources if isinstance(d,dict)]
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
    args=parser.parse_args()
    output=args.output or ("eval/retrieval_metrics.jsonl" if args.mode=="retrieval"
                           else "eval/gpt_judge_scores.jsonl")
    print("New rows:",benchmark(args.input,output,args.mode))

if __name__=="__main__": main()
