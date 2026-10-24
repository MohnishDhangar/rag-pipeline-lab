"""Local Streamlit demo; never deploy with a shared user's OAuth credential.

Start: streamlit run dashboard/app.py
"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from rag_lab.evaluation.benchmark import read_jsonl

st.set_page_config(page_title="RAG Pipeline Lab",layout="wide")
st.title("RAG Pipeline Lab")
st.caption("Adaptive scientific retrieval, source evidence, and controlled evaluation")

tab_live,tab_results,tab_sources=st.tabs(["Ask research questions","Benchmark explorer","Evidence graph"])

with tab_live:
    st.write("Uses local OAuth credentials and your selected ChatGPT account model.")
    q=st.text_area("Scientific question",placeholder="Compare PINNs and MeshGraphNets...")
    backend=st.selectbox("Retrieval backend",["dense","colbert"])
    strategy=st.selectbox("Strategy",["adaptive","naive","multi_query","rag_fusion",
                                     "hyde","step_back","decomposition"])
    use_graph=st.checkbox("Augment retrieval with scientific evidence graph",value=False)
    verify_flag=st.checkbox("GPT verify answers with uncertain evidence (extra inference)",value=False)
    if st.button("Generate answer",disabled=not q.strip()):
        try:
            from rag_lab.evaluation.generate import create_backend, make_strategy
            from rag_lab.llms.gpt_adapter import build_chat_model
            from rag_lab.routing.adaptive_router import AdaptiveRouter
            import os
            model=os.environ.get("RAG_GPT_MODEL")
            if not model: raise ValueError("Set RAG_GPT_MODEL after OAuth model discovery")
            store=create_backend(backend,Path("."))
            if use_graph:
                from rag_lab.retrieval.evidence_augmented import EvidenceAugmentedVectorStore
                from rag_lab.knowledge.evidence_graph import EvidenceGraph
                store=EvidenceAugmentedVectorStore(store,EvidenceGraph.load("data/evidence_graph.json"))
            llm=build_chat_model(model=model)
            if strategy=="adaptive":
                names=["naive","multi_query","rag_fusion","hyde","step_back","decomposition"]
                alternatives={n:make_strategy(n,store,llm) for n in names}
                path=Path("data/adaptive_router.joblib")
                runner=AdaptiveRouter(alternatives,path if path.exists() else None)
            else:
                runner=make_strategy(strategy,store,llm)
            from rag_lab.core.pipeline import EvidenceAwarePipeline
            result=EvidenceAwarePipeline(runner,verify_on_warning=verify_flag).run_with_trace(
                q,cross_document=("compare" in q.lower() or "contrast" in q.lower())
            )
            st.session_state["current_result"]=result.to_dict()
        except Exception as exc:
            st.error(type(exc).__name__+": "+str(exc))
    result=st.session_state.get("current_result")
    if result:
        st.subheader("Answer")
        st.write(result["answer"])
        st.caption("Selected strategy: "+str(result.get("trace",{}).get("selected_route",result["strategy"])))
        st.write("Latency (ms):",result["latency_ms"])
        st.write("Reliability:",result.get("trace",{}).get("reliability"))
        with st.expander("Retrieved source evidence"):
            for src,context in zip(result["sources"],result["contexts"]):
                st.write(src)
                st.text(context[:2200])

with tab_results:
    p=Path("eval")
    files=sorted(p.glob("*.jsonl")) if p.exists() else []
    result_file=st.selectbox("Saved evaluation file",files,format_func=str) if files else None
    if result_file:
        rows=read_jsonl(result_file)
        if rows:
            frame=pd.DataFrame(rows)
            numeric=[k for k in ("faithfulness","answer_correctness","answer_relevance",
                                  "citation_support","precision_at_k","recall_at_k","mrr_at_k","ndcg_at_k")
                     if k in frame.columns]
            dimensions=[k for k in ("strategy","backend") if k in frame.columns]
            st.dataframe(frame, use_container_width=True)
            if numeric and dimensions:
                st.subheader("Mean benchmark metrics")
                means=frame.groupby(dimensions,dropna=False)[numeric].mean(numeric_only=True)
                st.dataframe(means.round(3),use_container_width=True)
                st.bar_chart(means)
            else:
                st.info("These are generation traces; use the offline evaluators for quality scores.")
        else: st.info("No rows found.")
    else: st.info("No local JSONL runs found. Use the generation and evaluation CLI.")

with tab_sources:
    graph_path=Path("data/evidence_graph.json")
    if not graph_path.exists():
        st.info("Build the graph first with: python -m rag_lab.knowledge.build_graph")
    else:
        from rag_lab.knowledge.evidence_graph import EvidenceGraph
        graph=EvidenceGraph.load(graph_path)
        st.metric("Indexed evidence passages",len(graph.nodes))
        st.metric("Scientific papers",len(graph.papers))
        query=st.text_input("Explore concept connections")
        if query:
            for evidence in graph.query(query):
                st.markdown("**"+str(evidence.paper_id)+" · page "+str(evidence.page)+"**")
                st.write(evidence.text[:1400])
