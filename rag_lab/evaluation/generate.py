"""Resumable GPT-only generation across existing RAG strategies and backends.

Requires already-built local Chroma and/or PyLate indexes, and GOOGLE_API_KEY
for Gemini query embeddings. Run after ChatGPT OAuth login and models selection.
"""
import argparse, json, os
from pathlib import Path

from rag_lab.llms.gpt_adapter import build_chat_model, InferenceError
from rag_lab.utils import append_jsonl
from rag_lab.evaluation.benchmark import read_jsonl
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.provenance import ProvenanceStore
from rag_lab.retrieval.adaptive_store import AdaptiveRetrievalStore
from rag_lab.retrieval.query_router import classify_question, VERSION as ROUTER_VERSION
from rag_lab.routing.controller_v2 import VERSION as CONTROLLER_V2_VERSION
from rag_lab.retrieval.backend_factory import create_backend


def make_strategy(name, backend, llm, registry=None, *, crag_max_grades=2, crag_pool_k=28):
    from rag_lab.strategies.naive import NaiveStrategy
    from rag_lab.strategies.multi_query import MultiQueryStrategy
    from rag_lab.strategies.rag_fusion import RagFusionStrategy
    from rag_lab.strategies.hyde import HydeStrategy
    from rag_lab.strategies.step_back import StepBackStrategy
    from rag_lab.strategies.decomposition import DecompositionStrategy
    from rag_lab.strategies.adaptive import AdaptiveNaiveStrategy
    from rag_lab.strategies.crag import CRAGStrategy
    if name == "crag":
        return CRAGStrategy(backend,llm,registry=registry,
                            max_grades=crag_max_grades,pool_k=crag_pool_k)
    registry=dict(naive=NaiveStrategy,multi_query=MultiQueryStrategy,
                  rag_fusion=RagFusionStrategy,hyde=HydeStrategy,
                  step_back=StepBackStrategy,decomposition=DecompositionStrategy,
                  adaptive=AdaptiveNaiveStrategy)
    return registry[name](backend,llm=llm)

def usable_generation(row):
    # An explicitly recorded CRAG abstention is a valid experimental outcome,
    # not a generation failure. It MUST remain counted on resumed runs.
    if row.get("generation_status") == "abstained":
        trace=(row.get("trace") or {}).get("crag") or {}
        return (row.get("strategy") == "crag"
                and trace.get("action") == "abstain"
                and row.get("answer") == "Insufficient retrieved evidence to answer this question.")
    return bool(row.get("answer") and row.get("contexts")
                and row.get("answer") != "Insufficient retrieved evidence to answer this question.")


def generate(root, results, strategies, backends, model=None, limit=None, graph_path=None,
             verify_on_warning=False, chroma_path=None, chroma_collection=None,
             adaptive_k=4, adaptive_fetch_k=20, adaptive_lambda=.5, ids=None,
             crag_max_grades=2, crag_pool_k=28, controller_grader_budget=1,
             controller_require_colbert=False):
    if "adaptive_v2" in strategies and (list(backends)!=["dense"] or graph_path):
        raise ValueError("adaptive_v2 runs once under --backends dense without graph")
    if crag_max_grades not in (1,2) or controller_grader_budget not in (0,1,2):
        raise ValueError("CRAG grades must be 1-2; controller budget must be 0-2")
    dataset=json.loads((root/"eval"/"eval_set.json").read_text())
    questions=dataset["questions"]
    registry = PaperRegistry.from_eval(dataset)
    if ids is not None:
        requested=set(ids)
        unknown=requested-{q["id"] for q in questions}
        if unknown:
            raise ValueError(f"Unknown evaluation IDs: {sorted(unknown)}")
        questions=[q for q in questions if q["id"] in requested]
    if limit is not None: questions=questions[:limit]
    adaptive_config = {"k": adaptive_k, "fetch_k": adaptive_fetch_k,
                       "lambda_mult": adaptive_lambda, "router_version": ROUTER_VERSION}
    config_key = lambda name, row: (
        json.dumps(row.get("retrieval_config"), sort_keys=True)
        if name in ("adaptive","crag","adaptive_v2") else None
    )
    crag_config={"version":"closed-corpus-crag-v5",
                 "max_grades":crag_max_grades,"pool_k":crag_pool_k}
    controller_config={"version":CONTROLLER_V2_VERSION,
                       "max_grader_calls":controller_grader_budget,
                       "crag_pool_k":crag_pool_k,
                       "require_colbert":controller_require_colbert}
    done = {(r.get("strategy"), r.get("backend"), r.get("id"),
             config_key(r.get("strategy"), r))
            for r in read_jsonl(results) if usable_generation(r)}
    llm=build_chat_model(model=model)
    count=0
    for backend_name in backends:
        store=create_backend(backend_name,root,chroma_path,chroma_collection)
        if graph_path:
            from rag_lab.retrieval.evidence_augmented import EvidenceAugmentedVectorStore
            from rag_lab.knowledge.evidence_graph import EvidenceGraph
            store=EvidenceAugmentedVectorStore(store,EvidenceGraph.load(graph_path))
        # Annotate sources BEFORE GPT; registry never sees question labels.
        raw_store = store
        store = ProvenanceStore(raw_store, registry)
        backend_key=backend_name+"+graph" if graph_path else backend_name
        for name in strategies:
            selected = store
            if name == "adaptive":
                if backend_name != "dense" or graph_path:
                    raise ValueError("Adaptive retrieval currently requires the plain dense Chroma backend")
                selected = ProvenanceStore(
                    AdaptiveRetrievalStore(raw_store, k=adaptive_k,
                                           fetch_k=adaptive_fetch_k,
                                           lambda_mult=adaptive_lambda), registry)
            config=(adaptive_config if name=="adaptive"
                    else crag_config if name=="crag"
                    else controller_config if name=="adaptive_v2"
                    else None)
            if name=="adaptive_v2":
                from rag_lab.strategies.adaptive_v2 import AdaptiveV2Strategy
                pools={"dense":store}
                folder=root/"data"/"colbert_index"
                if folder.is_dir() and any(folder.iterdir()):
                    # If index exists but is corrupt, fail loudly instead of
                    # silently using a different arm of the experiment.
                    pools["colbert"]=ProvenanceStore(create_backend("colbert",root),registry)
                elif controller_require_colbert:
                    raise FileNotFoundError("adaptive_v2 requires ColBERT index in data/colbert_index")
                strategy=AdaptiveV2Strategy(
                    pools,llm,registry,max_grader_calls=controller_grader_budget,
                    crag_pool_k=crag_pool_k)
                config={**controller_config,"colbert_available":"colbert" in pools}
            else:
                strategy=make_strategy(name,selected,llm,registry,
                                       crag_max_grades=crag_max_grades,
                                       crag_pool_k=crag_pool_k)
            for q in questions:
                key=(name,backend_key,q["id"],
                     json.dumps(config,sort_keys=True) if config is not None else None)
                if key in done: continue
                try:
                    from rag_lab.core.pipeline import EvidenceAwarePipeline
                    # Do not feed the gold question_type to the runtime pipeline.
                    predicted = classify_question(q["question"])
                    result=EvidenceAwarePipeline(strategy,verify_on_warning=verify_on_warning).run_with_trace(
                        q["question"],cross_document=predicted.question_type=="cross_document"
                    )
                    # Deliberate CRAG abstention is part of the experiment.
                    # Save it, count it, and continue to all remaining arms.
                    crag_trace=(result.trace or {}).get("crag") or {}
                    abstained=(name=="crag" and crag_trace.get("action")=="abstain"
                               and result.answer ==
                               "Insufficient retrieved evidence to answer this question.")
                    if not result.contexts and not abstained:
                        print(f"Stopping: {backend_key}/{name}/{q['id']} retrieved no evidence. "
                              "No successful result will be recorded.")
                        return count
                    # Every strategy now sees canonical passage metadata before GPT.
                    unresolved=sum(not src.get("paper_id") for src in result.sources)
                    if unresolved:
                        print(f"WARNING: {unresolved} source chunks have unresolved paper IDs "
                              f"for {q['id']}; strict retrieval scores will be unscored.")
                    row={**result.to_dict(), "id":q["id"],
                         "strategy":name,"backend":backend_key,
                         "question_type":q.get("question_type"),
                         "ground_truth":q.get("ground_truth"),
                         "source_documents":q.get("source_documents"),
                         "model":model or llm.model_name,
                         "runtime_question_type":predicted.question_type,
                         "generation_status":"abstained" if abstained else "answered",
                         **({"retrieval_config":config} if config is not None else {})}
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
    p.add_argument("--ids",nargs="+",help="Select specific evaluation IDs for matched paired generation")
    p.add_argument("--chroma-path",type=Path,help="Existing, nonempty Chroma index")
    p.add_argument("--chroma-collection",help="Collection name (default: langchain)")
    p.add_argument("--graph",type=Path,help="Optional built scientific evidence graph")
    p.add_argument("--verify-on-warning",action="store_true",help="Spend one additional GPT call for uncertain answers")
    p.add_argument("--crag-max-grades",type=int,choices=[1,2],default=2)
    p.add_argument("--crag-pool-k",type=int,default=28)
    p.add_argument("--controller-grader-budget",type=int,choices=[0,1,2],default=1)
    p.add_argument("--controller-require-colbert",action="store_true")
    p.add_argument("--adaptive-k",type=int,default=4)
    p.add_argument("--adaptive-fetch-k",type=int,default=20)
    p.add_argument("--adaptive-lambda",type=float,default=.5,
                   help="MMR relevance weight for adaptive cross-paper questions")
    a=p.parse_args()
    if a.adaptive_k < 1 or a.adaptive_fetch_k < a.adaptive_k or not 0 <= a.adaptive_lambda <= 1:
        p.error("require adaptive_k >= 1, adaptive_fetch_k >= adaptive_k, lambda in [0,1]")
    print("Generated:",generate(a.root,a.output,a.strategies,a.backends,a.model,a.limit,a.graph,
                                a.verify_on_warning,a.chroma_path,a.chroma_collection,
                                a.adaptive_k,a.adaptive_fetch_k,a.adaptive_lambda,a.ids,
                                a.crag_max_grades,a.crag_pool_k,
                                a.controller_grader_budget,a.controller_require_colbert))

if __name__=="__main__": main()
