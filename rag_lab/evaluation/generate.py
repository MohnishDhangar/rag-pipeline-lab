"""Resumable GPT-only generation across existing RAG strategies and backends.

Requires already-built local Chroma and/or PyLate indexes, and GOOGLE_API_KEY
for Gemini query embeddings. Run after ChatGPT OAuth login and models selection.
"""
import argparse, json
from pathlib import Path

from rag_lab.llms.gpt_adapter import build_chat_model, InferenceError
from rag_lab.utils import append_jsonl
from rag_lab.evaluation.benchmark import read_jsonl

def create_backend(name, root):
    if name=="dense":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_chroma import Chroma
        embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return Chroma(embedding_function=embeddings,
                      persist_directory=str(root/"data"/"chroma_naive_gemini"))
    from rag_lab.retrieval.colbert_backend import ColBERTVectorStoreShim
    with (root/"data"/"id_to_text.json").open() as f: mapping=json.load(f)
    return ColBERTVectorStoreShim(index_folder=str(root/"data"/"colbert_index"),
                                  id_to_text=mapping)

def make_strategy(name, backend, llm):
    from rag_lab.strategies.naive import NaiveStrategy
    from rag_lab.strategies.multi_query import MultiQueryStrategy
    from rag_lab.strategies.rag_fusion import RagFusionStrategy
    from rag_lab.strategies.hyde import HydeStrategy
    from rag_lab.strategies.step_back import StepBackStrategy
    from rag_lab.strategies.decomposition import DecompositionStrategy
    registry=dict(naive=NaiveStrategy,multi_query=MultiQueryStrategy,
                  rag_fusion=RagFusionStrategy,hyde=HydeStrategy,
                  step_back=StepBackStrategy,decomposition=DecompositionStrategy)
    return registry[name](backend,llm=llm)

def generate(root, results, strategies, backends, model=None, limit=None):
    questions=json.loads((root/"eval"/"eval_set.json").read_text())["questions"]
    if limit is not None: questions=questions[:limit]
    done={(r.get("strategy"),r.get("backend"),r.get("id"))
          for r in read_jsonl(results) if r.get("answer")}
    llm=build_chat_model(model=model)
    count=0
    for backend_name in backends:
        store=create_backend(backend_name,root)
        for name in strategies:
            strategy=make_strategy(name,store,llm)
            for q in questions:
                key=(name,backend_name,q["id"])
                if key in done: continue
                try:
                    result=strategy.run_with_trace(q["question"])
                    row={**result.to_dict(), "id":q["id"],
                         "strategy":name,"backend":backend_name,
                         "question_type":q.get("question_type"),
                         "ground_truth":q.get("ground_truth"),
                         "source_documents":q.get("source_documents"),
                         "model":model or llm.model_name}
                    append_jsonl(results,row)
                    done.add(key)
                    count+=1
                except (InferenceError, ConnectionError, TimeoutError) as exc:
                    print("Inference stopped without marking this row complete:",exc)
                    return count
    return count

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--output",type=Path,default=Path("eval/generated_gpt.jsonl"))
    p.add_argument("--model")
    p.add_argument("--strategies",nargs="+",default=["naive","multi_query","rag_fusion","hyde","step_back","decomposition"])
    p.add_argument("--backends",nargs="+",default=["dense"],choices=["dense","colbert"])
    p.add_argument("--limit",type=int)
    a=p.parse_args()
    print("Generated:",generate(a.root,a.output,a.strategies,a.backends,a.model,a.limit))

if __name__=="__main__": main()
