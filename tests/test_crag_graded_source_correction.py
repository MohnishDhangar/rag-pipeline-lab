"""Test an explicitly named missing paper even when its initial passage failed grading."""
from langchain_core.documents import Document
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.provenance import ProvenanceStore
from rag_lab.strategies.crag import CRAGStrategy
from rag_lab.llms.gpt_adapter import build_chat_model


def doc(source,page,content):
    return Document(page_content=content,metadata={"source":source,"page":page})


class Store:
    def __init__(self):
        self.a=doc("ntk.pdf",0,"NTK convergence rates differ by loss component.")
        self.b_bad=doc("original.pdf",0,"Introduction to physics-informed models.")
        self.b_good=doc("original.pdf",1,
                        "Original PINNs use PDE loss as regularization and L-BFGS optimization.")
        self.calls=[]
    def as_retriever(self,search_kwargs=None):
        class Ret:
            def __init__(self,docs):self.docs=docs
            def invoke(self,query,config=None,**kwargs):return self.docs
        return Ret([self.a,self.b_bad])
    def get(self,include=None):
        assert include==["metadatas"]
        return {"metadatas":[x.metadata for x in (self.a,self.b_bad,self.b_good)]}
    def similarity_search(self,query,k=4,**kwargs):
        filter=kwargs.get("filter")
        self.calls.append((k,filter))
        if filter:
            source=filter["source"]
            return [d for d in [self.a,self.b_bad,self.b_good]
                    if d.metadata["source"]==source][:k]
        return [self.a,self.b_bad,self.b_good][:k]


class Transport:
    calls=0
    def generate(self,input,instructions=None):
        self.calls+=1
        return "Both papers differ [SOURCE ntk_pinn:pdfpage1]",{}


def relevance(q,docs):
    return [{
        "index":i,
        "grade":"incorrect" if d.page_content.startswith("Introduction") else "correct",
        "quote":"" if d.page_content.startswith("Introduction") else
                d.page_content[:30],
        "reason":"generic abstract" if d.page_content.startswith("Introduction") else "specific"
    } for i,d in enumerate(docs)]


def test_grader_forces_second_source_even_when_paper_already_present():
    registry=PaperRegistry([
        {"short_id":"ntk_pinn","file":"ntk.pdf"},
        {"short_id":"orig_pinn","file":"original.pdf"}])
    store=Store()
    backend=ProvenanceStore(store,registry)
    llm=build_chat_model(model="mock",transport=Transport())
    strategy=CRAGStrategy(backend,llm,registry=registry,
                          grader_fn=relevance,max_grades=2,k=4,pool_k=8)
    out=strategy.run_with_trace("Both NTK-PINN and Raissi et al. explain training difficulties.")
    assert out.trace["crag"]["action"]=="corrected"
    assert out.trace["crag"]["correction_retrieval"]["forced_paper_retries"]==["orig_pinn"]
    assert {s["paper_id"] for s in out.sources}=={"ntk_pinn","orig_pinn"}
    assert "Original PINNs use PDE" in " ".join(out.contexts)
    assert len(out.trace["crag"]["initial_grades"])==2
    assert any(g["paper_id"]=="orig_pinn" and g["grade"]=="correct"
               for g in out.trace["crag"]["corrective_grades"])
    assert any(call[1]=={"source":"original.pdf"} for call in store.calls)
