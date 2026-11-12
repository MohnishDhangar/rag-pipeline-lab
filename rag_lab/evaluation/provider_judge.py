"""Provider-aware, manifest-bound judging for both repository architectures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from rag_lab.evaluation.jsonl_io import read_jsonl
from rag_lab.evaluation.identity import row_fingerprint, RUBRIC_VERSION
from rag_lab.evaluation.judge import evaluate_row
from rag_lab.llms.factory import build_transport, judge_selection, inference_identity
from rag_lab.llms.gpt_adapter import InferenceError
from rag_lab.utils import append_jsonl

ABSTAIN = "Insufficient retrieved evidence to answer this question."
UNVERIFIED = "I could not verify the generated answer from the retrieved evidence."

def eligible(row):
    return bool(row.get("answer") and row.get("contexts")
                and row.get("generation_status") != "abstained"
                and row.get("answer") not in (ABSTAIN, UNVERIFIED))

def row_key(row):
    return (row["strategy"], row["backend"], row["id"], row_fingerprint(row))

def judge_file(input_path, output_path, *, provider=None, model=None, transport=None):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Judge output must differ from generation input")
    provider, model = judge_selection(provider, model)
    identity = inference_identity(provider, model)
    rows = read_jsonl(input_path)
    keys = [row_key(r) for r in rows]
    if len({k[:3] for k in keys}) != len(keys):
        raise ValueError("Duplicate generation arm/question; split incompatible experiments")
    if not rows:
        raise ValueError("No generation rows to judge")
    config = {"schema": "provider-judge-v1", "rubric": RUBRIC_VERSION,
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "inference": identity}
    sidecar = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != config:
            raise ValueError("Judge experiment mismatch; use a new output filename")
    else:
        if output_path.exists() and output_path.stat().st_size:
            raise ValueError("Judge output has no manifest; use a new filename")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("x", encoding="utf-8") as f:
            json.dump(config,f,indent=2)
            f.write("\n")
    done=set()
    valid={row_key(r) for r in rows if eligible(r)}
    for r in read_jsonl(output_path):
        key=(r["strategy"],r["backend"],r["id"],r["input_fingerprint"])
        if (key not in valid or key in done or r.get("judge_inference") != identity
                or r.get("judge_version") != RUBRIC_VERSION):
            raise ValueError("Invalid or incompatible judge checkpoint")
        done.add(key)
    pending=[r for r in rows if eligible(r) and row_key(r) not in done]
    if not pending:
        return 0
    transport=transport or build_transport(provider,model)
    if getattr(transport,"model",None)!=model or getattr(transport,"provider","chatgpt_oauth")!=provider:
        raise ValueError("Judge transport does not match declared provider/model")
    count=0
    for r in pending:
        key=row_key(r)
        try:
            scores=evaluate_row(r,transport)
        except (InferenceError, ConnectionError, TimeoutError) as exc:
            print(f"Judge interrupted at {key[:3]}; {count} new rows preserved: {exc}")
            break
        append_jsonl(output_path,{"strategy":key[0],"backend":key[1],"id":key[2],
                                 "input_fingerprint":key[3],"judge_inference":identity,**scores})
        count+=1
    return count

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--judge-provider",choices=("chatgpt_oauth","groq","gemini"))
    p.add_argument("--judge-model")
    a=p.parse_args()
    print("New judged rows:",judge_file(a.input,a.output,provider=a.judge_provider,model=a.judge_model))

if __name__=="__main__":
    main()
