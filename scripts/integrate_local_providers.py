"""Port provider integration onto the newer local manifest-bound checkpoint.

Run from repository root. --check reports applicability; --apply edits only
recognized source anchors. No commit, merge, reset or network request.
"""
import argparse
from pathlib import Path

def replace(text, old, new):
    if new in text:
        return text
    if text.count(old)!=1:
        raise ValueError("Expected exactly one source anchor: "+old[:100])
    return text.replace(old,new,1)

def plan(root):
    changes={}
    path=root/"rag_lab/evaluation/generate.py"
    text=path.read_text()
    if not (root/"rag_lab/core/factory.py").is_file() or "experiment_id = bind(results, experiment)" not in text:
        raise ValueError("Not the local manifest-bound refactor; use normal Git integration.")
    new=replace(text,"from rag_lab.llms.gpt_adapter import build_chat_model, InferenceError",
                "from rag_lab.llms.gpt_adapter import InferenceError\nfrom rag_lab.llms.factory import build_chat_model, resolve_provider, inference_identity, usage_snapshot, usage_delta")
    new=replace(new,"registry_path=None, frozen_eval=None):","registry_path=None, frozen_eval=None, provider=None):")
    new=replace(new,"    llm = build_chat_model(model=model)",
                "    selected_provider = resolve_provider(provider)\n    llm = build_chat_model(model=model, provider=provider) if provider else build_chat_model(model=model)")
    new=replace(new,"    config['frozen_eval_sha256']=",
                "    config['inference'] = inference_identity(selected_provider, config['model'])\n    config['frozen_eval_sha256']=")
    new=replace(new,"                predicted = classify_question(q[\'question\'])",
                "                before = usage_snapshot(llm)\n                predicted = classify_question(q[\'question\'])")
    new=replace(new,"'runtime_question_type': predicted.question_type,",
                "'runtime_question_type': predicted.question_type,\n                       'provider': selected_provider, 'inference': config['inference'],\n                       'inference_usage': usage_delta(llm, before),")
    new=replace(new,'    p.add_argument("--model")','    p.add_argument("--model")\n    p.add_argument("--provider", choices=("chatgpt_oauth","groq","gemini"))')
    new=replace(new,"a.eval_set, a.router_model, a.registry, a.frozen_eval))",
                "a.eval_set, a.router_model, a.registry, a.frozen_eval, provider=a.provider))")
    compile(new,str(path),"exec")
    changes[path]=(text,new)
    # The local shared pipeline already injects the selected transport. Preserve it.
    # Enable provider selection on local dashboard using environment defaults;
    # retain its local UI, settings and shared factory.
    path=root/"dashboard/app.py"
    if path.exists():
        text=path.read_text()
        new=text.replace("from rag_lab.llms.gpt_adapter import build_chat_model","from rag_lab.llms.factory import build_chat_model")
        if new!=text:
            compile(new,str(path),"exec")
            changes[path]=(text,new)
    # A local judge may contain other changes. Modify only the existing
    # transport fallback, never copy a remote evaluator over the local file.
    path=root/"rag_lab/evaluation/judge.py"
    text=path.read_text()
    new=text
    if "llm = transport or GPTTransport()" in text:
        new=replace(new,"llm = transport or GPTTransport()","llm = transport or build_transport(*judge_selection())")
        new=replace(new,"from rag_lab.llms.gpt_adapter import GPTTransport, parse_object",
                    "from rag_lab.llms.gpt_adapter import parse_object\nfrom rag_lab.llms.factory import build_transport, judge_selection")
        new=replace(new,'"model":llm.model,','"model":llm.model, "judge_provider":getattr(llm,"provider","chatgpt_oauth"),')
    elif "build_transport(*judge_selection())" not in text:
        raise ValueError("Local judge has changed; inspect its transport fallback manually.")
    if "class JudgeScores(BaseModel):" not in new:
        new=replace(new,"import json\n","import json\nfrom pydantic import BaseModel, Field\n")
        new=replace(new,"def evaluate_row(row, transport=None):",
                    "class JudgeScores(BaseModel):\n    faithfulness: float = Field(ge=0, le=1)\n    answer_correctness: float = Field(ge=0, le=1)\n    answer_relevance: float = Field(ge=0, le=1)\n    citation_support: float = Field(ge=0, le=1)\n    reason: str\n\ndef evaluate_row(row, transport=None):")
        new=replace(new,'    text, usage = llm.generate([{"role":"user","content":content}],instructions=SYSTEM)',
                    '    items=[{"role":"user","content":content}]\n    if hasattr(llm,"generate_structured"):\n        text, usage=llm.generate_structured(items,SYSTEM,JudgeScores)\n    else:\n        text, usage=llm.generate(items,instructions=SYSTEM)')
    compile(new,str(path),"exec")
    changes[path]=(text,new)
    path=root/"rag_lab/evaluation/identity.py"
    text=path.read_text()
    new=replace(text,'    data=json.dumps(content,ensure_ascii=False,sort_keys=True)',
                '    if row.get("inference") is not None:\n        content["inference"]=row["inference"]\n    data=json.dumps(content,ensure_ascii=False,sort_keys=True)')
    compile(new,str(path),"exec")
    changes[path]=(text,new)
    return changes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",action="store_true")
    group.add_argument("--apply",action="store_true")
    p.add_argument("--root",type=Path,default=Path("."))
    a=p.parse_args()
    changes=plan(a.root)
    for path,(old,new) in changes.items():
        if old!=new:
            print(("Updating " if a.apply else "Would update ")+str(path))
            if a.apply:
                path.write_text(new)
    print("Use a NEW generation output and rag_lab.evaluation.provider_judge for manifest-bound judging.")
    print("Review git diff and run pytest before live calls. Dashboard may need explicit UI model cleanup.")

if __name__=="__main__":
    main()
