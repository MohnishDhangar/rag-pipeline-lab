"""Offline preflight for controlled Dense/ColBERT + CRAG GPT pilot.

Does NOT perform network operations, load the PyLate model, or read tokens.
Checks prerequisites in the selected Python environment before spending
Gemini embeddings/GPT model requests.
"""
from __future__ import annotations
import argparse,importlib.util,json,os
from pathlib import Path

DEFAULT_IDS=("b1_gns_03","b1_cross_01","b2_cross_003")


def verify(root, *, ids=DEFAULT_IDS, expected_chunks=576):
    from rag_lab.retrieval.chroma_health import require_populated_index
    from rag_lab.retrieval.colbert_health import check
    root=Path(root)
    corpus=json.loads((root/"eval"/"eval_set.json").read_text(encoding="utf-8"))
    questions={q["id"]:q for q in corpus["questions"]}
    absent=set(ids)-set(questions)
    if absent:raise ValueError(f"Unknown pilot questions: {sorted(absent)}")
    counts={}
    location=Path(os.environ.get("RAG_CHROMA_PATH")
                  or (root/"data"/"chroma_naive_gemini"))
    collection=os.environ.get("RAG_CHROMA_COLLECTION","langchain")
    counts["chroma"]=require_populated_index(location,collection)
    if counts["chroma"]!=expected_chunks:
        raise ValueError(f"Chroma has {counts['chroma']} passages; expected {expected_chunks}")
    colbert=check(root,expected_chunks=expected_chunks)
    if importlib.util.find_spec("pylate") is None:
        raise RuntimeError("PyLate unavailable in this interpreter")
    if importlib.util.find_spec("jwt") is None:
        raise RuntimeError('GPT OAuth requires PyJWT: python -m pip install "PyJWT[crypto]>=2.8"')
    if importlib.util.find_spec("langchain_classic") is None:
        raise RuntimeError('Install strategy dependencies: pip install -e ".[rag,dev]"')
    if importlib.util.find_spec("langchain_google_genai") is None:
        raise RuntimeError('Install dense retrieval dependencies: pip install -e ".[rag,dev]"')
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError("Dense retrieval needs GOOGLE_API_KEY or GEMINI_API_KEY")
    if not os.environ.get("RAG_GPT_MODEL"):
        raise RuntimeError("Set RAG_GPT_MODEL to a model listed by rag_lab.llms.cli models")
    return {
        "ready_for_local_live_pilot":True,
        "index":{"chroma_chunks":counts["chroma"],
                 "colbert_mapping_chunks":colbert["mapping_chunks"],
                 "colbert_manifest":colbert["manifest"]["status"]},
        "questions":[{"id":qid,"text":questions[qid]["question"]}
                     for qid in ids],
        "expected_arms":4*len(ids),
        "network_requests":0,
        "credentials_checked":"environment variable presence only",
        "warning":"Real PyLate output and GPT access are NOT tested by preflight."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--expected-chunks",type=int,default=576)
    p.add_argument("--ids",nargs="+",default=list(DEFAULT_IDS))
    a=p.parse_args()
    print(json.dumps(verify(a.root,ids=a.ids,
                            expected_chunks=a.expected_chunks),indent=2))


if __name__=="__main__":main()
