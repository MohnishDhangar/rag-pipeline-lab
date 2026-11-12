"""Resumable offline retrieval-only and GPT judge evaluation.

python -m rag_lab.evaluation.benchmark retrieval --input eval/generated_gpt.jsonl
python -m rag_lab.evaluation.benchmark judge --input eval/generated_gpt.jsonl
"""
import argparse, json, hashlib
from pathlib import Path

from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.evaluation.judge import evaluate_row
from rag_lab.evaluation.identity import row_fingerprint, RUBRIC_VERSION
from rag_lab.utils import append_jsonl
from rag_lab.llms.gpt_adapter import InferenceError
from rag_lab.llms.factory import build_transport, judge_selection, inference_identity

def read_jsonl(path):
    if not Path(path).exists(): return []
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
    return rows

RETRIEVAL_VERSION = "retrieval-metrics-v2"



def benchmark(input_path, output_path, mode, registry=None, *, judge_provider=None, judge_model=None):
    transport=None
    judge_identity=None
    if mode == 'judge':
        selected, model=judge_selection(judge_provider,judge_model)
        judge_identity=inference_identity(selected,model)
        if any(r.get('judge_inference')!=judge_identity for r in read_jsonl(output_path)):
            raise ValueError('Judge inference configuration mismatch; use a new output filename')
        transport=build_transport(selected,model)
    completed = {
        (r.get("strategy"),r.get("backend"),r.get("id"),
         r.get("judge_version") if mode=="judge" else r.get("metric_version"),
         r.get("input_fingerprint"))
        for r in read_jsonl(output_path)
    }
    successful = 0
    for row in read_jsonl(input_path):
        if not row.get("answer") or not row.get("contexts"): continue  # no GPT judging on missing evidence
        signature=row_fingerprint(row)
        key=(row.get("strategy"),row.get("backend"),row.get("id"),
             RUBRIC_VERSION if mode=="judge" else RETRIEVAL_VERSION, signature)
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
            try: scores=evaluate_row(row,transport)
            except (InferenceError, ConnectionError, TimeoutError) as exc:
                print("Judge stopped; checkpoint preserved:",type(exc).__name__,str(exc))
                break
        append_jsonl(output_path, {"strategy":row.get("strategy"),"backend":row.get("backend"),
                                    "id":row.get("id"),"metric_version":RETRIEVAL_VERSION if mode=="retrieval" else None,
                                    "input_fingerprint":signature,**({"judge_inference":judge_identity} if judge_identity else {}),**scores})
        completed.add(key)
        successful+=1
    return successful

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["retrieval","judge"])
    parser.add_argument("--input",default="eval/generated_gpt.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--judge-provider", choices=("chatgpt_oauth","groq","gemini"))
    parser.add_argument("--judge-model")
    parser.add_argument("--eval",default="eval/eval_set.json",help="Ground-truth registry (no retrieval leakage)")
    args=parser.parse_args()
    output=args.output or ("eval/retrieval_metrics_v2.jsonl" if args.mode=="retrieval"
                           else "eval/gpt_judge_scores_v2.jsonl")
    registry=PaperRegistry.from_eval(json.loads(Path(args.eval).read_text()))
    print("New rows:",benchmark(args.input,output,args.mode,registry,judge_provider=args.judge_provider,judge_model=args.judge_model))

if __name__=="__main__": main()
