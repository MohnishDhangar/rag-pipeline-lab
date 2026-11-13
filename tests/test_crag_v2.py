import pytest
from langchain_core.documents import Document
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.strategies.crag import CRAGStrategy,ABSTAIN


def doc(p,text="relevant evidence"):
    return Document(page_content=text,metadata={"source":p+".pdf","page":1})


class Store:
    def __init__(self,initial,repaired=None):
        self.initial,self.repaired=initial,repaired or initial
    def as_retriever(self,search_kwargs=None):
        self.k=search_kwargs.get("k",4)
        return self
    def invoke(self,input,config=None,**kwargs):
        return self.initial
    def similarity_search(self,query,k=4,**kwargs):
        return self.repaired


class LLM:
    def __init__(self):
        self.calls=0
    def __or__(self,other):
        return self
    def invoke(self,inputs):
        self.calls+=1
        return "Answered [SOURCE xpinn.pdf:pdfpage2]"


def grader(q,documents):
    return [{"index":i,"grade":"correct","quote":"relevant evidence","reason":"direct"}
            for i,d in enumerate(documents)]


def registry():
    return PaperRegistry([{"short_id":"xpinn","file":"xpinn.pdf"},
                          {"short_id":"sk_pinn","file":"sk_pinn.pdf"}])


def test_crag_abstains_when_batch_reference_unresolved():
    llm=LLM()
    s=CRAGStrategy(Store([doc("xpinn")]),llm,registry=registry(),grader_fn=grader)
    r=s.run_with_trace("All three papers in this batch address efficiency.")
    assert r.answer==ABSTAIN and r.trace["crag"]["action"]=="abstain"
    assert r.trace["gpt_requests"]==0
    assert not r.contexts


def test_crag_corrects_missing_named_paper_with_real_sources():
    # LLM need not run: mocked structured grader plus the actual generation
    # pipeline is covered separately by integration tests.
    st=Store([doc("xpinn")],[doc("xpinn"),doc("sk_pinn")])
    class FakeCRAG(CRAGStrategy):
        def run(self,q):
            # Keep production's retrieval/grading/gating, but bypass final
            # LangChain prompt chaining with a fake compatible LLM.
            return super().run(q)
    s=CRAGStrategy(st,LLM(),registry=registry(),grader_fn=grader)
    # The source-aware stage must include both named papers, not arbitrary 2.
    docs,trace=s.source_retriever.retrieve("Both XPINN and SK-PINN improve training.",
                                         initial=[doc("xpinn")])
    assert {s.registry.resolve(d.metadata["source"]) for d in docs}=={"xpinn","sk_pinn"}
    assert not trace["missing_requested_papers"]


def test_invalid_fabricated_quote_cannot_be_accepted():
    assert not CRAGStrategy._accepted([doc("xpinn")],[
        type("Grade",(),{"grade":"correct","quote":"fabricated"})()
    ])
