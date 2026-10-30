"""CRAG v6: no LLM grader can delete initially useful evidence."""
from langchain_core.documents import Document
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.strategies.crag_v6 import select_preserving_initial,CRAGV6Strategy,ABSTAIN


def doc(source,text):
    return Document(page_content=text,metadata={"source":source,"page":0})


def registry():
    return PaperRegistry([
        {"short_id":"xpinn","file":"xpinn.pdf"},
        {"short_id":"sk_pinn","file":"sk_pinn.pdf"}])


def test_preserves_all_initial_passages_without_missing_sources():
    initial=[doc("xpinn.pdf",f"Evidence {i}") for i in range(4)]
    assert select_preserving_initial(initial,[],registry(),("xpinn",),4)==initial


def test_adds_missing_paper_without_losing_all_initial_context():
    initial=[doc("xpinn.pdf",f"XPINN {i}") for i in range(4)]
    selected=select_preserving_initial(
        initial,[doc("sk_pinn.pdf","SK-PINN derivatives")],
        registry(),("xpinn","sk_pinn"),4)
    assert len(selected)==4
    assert initial[0] in selected
    assert any(d.metadata["source"]=="sk_pinn.pdf" for d in selected)


def test_deduplicates_retrieved_evidence():
    initial=[doc("xpinn.pdf","Same evidence")]
    assert len(select_preserving_initial(
        initial,initial,registry(),(),4))==1


class Retriever:
    def __init__(self,docs):self.docs=docs
    def invoke(self,q,config=None,**kw):return self.docs[:4]


class Store:
    def __init__(self,docs):self.docs=docs
    def as_retriever(self,search_kwargs=None):return Retriever(self.docs)
    def similarity_search(self,q,k=4,**kw):
        filt=kw.get("filter")
        docs=self.docs
        if filt:docs=[d for d in docs if d.metadata["source"]==filt["source"]]
        return docs[:k]
    def get(self,include=None):return {"metadatas":[d.metadata for d in self.docs]}


def test_no_evidence_abstains_without_llm():
    s=CRAGV6Strategy(Store([]),None,registry=registry())
    result=s.run_with_trace("What is the result?")
    assert result.answer==ABSTAIN
    assert result.trace["crag_v6"]["action"]=="abstain"
