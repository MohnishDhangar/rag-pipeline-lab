"""Enrich existing completed JSONL runs WITHOUT rerunning GPT or changing answers.

python -m rag_lab.evaluation.provenance_repair \
   --input eval/generated_gpt_repaired.jsonl \
   --output eval/generated_gpt_provenance.jsonl

The original answers' legacy citation syntax is preserved. For future runs,
new source IDs are unique passage digests and PDF page numbers are explicit.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.evaluation.benchmark import read_jsonl


def repair_row(row, registry):
    row=dict(row)
    sources=row.get("sources") or []
    contexts=row.get("contexts") or []
    if len(sources) != len(contexts):
        raise ValueError("Source/context count mismatch in question "+str(row.get("id")))
    fixed=[]
    for source,context in zip(sources,contexts):
        old=dict(source)
        new=registry.enrich(old,context)
        # Existing citations refer to the old PDF filename and zero-based page;
        # keep that exact reference for audit, never rewrite generated answers.
        new["legacy_id"]=old.get("id")
        fixed.append(new)
    row["sources"]=fixed
    row["provenance_schema"]="v2"
    return row


def repair_file(input_path,output_path,dataset_path):
    src=Path(input_path).resolve()
    dst=Path(output_path).resolve()
    if src == dst: raise ValueError("Never overwrite original benchmark outputs")
    if dst.exists(): raise FileExistsError(str(dst))
    data=json.loads(Path(dataset_path).read_text())
    reg=PaperRegistry.from_eval(data)
    rows=read_jsonl(src)
    report=Counter()
    with dst.open("w",encoding="utf-8") as f:
        for row in rows:
            updated=repair_row(row,reg)
            report["questions"]+=1
            for source in updated.get("sources",[]):
                report["passages"]+=1
                if source.get("paper_id"): report["resolved"]+=1
                else: report["unknown"]+=1
            f.write(json.dumps(updated,ensure_ascii=False)+"\n")
    return dict(report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--eval",default="eval/eval_set.json")
    a=parser.parse_args()
    print(json.dumps(repair_file(a.input,a.output,a.eval),indent=2))


if __name__=="__main__":main()
