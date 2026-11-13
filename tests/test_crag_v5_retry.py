"""Regression: a direct answer-bearing GNS hit is not discarded due to a bad quote."""
from langchain_core.documents import Document
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.provenance import ProvenanceStore
from rag_lab.strategies.crag import CRAGStrategy


class FakeRetriever:
    def __init__(self,docs):
        self.docs=docs
    def invoke(self,q,config=None,**kwargs):
        return self.docs


class FakeStore:
    def __init__(self):
        self.main=Document(page_content=(
            "Training noise mitigates accumulated errors during long rollouts "
            "by matching the distribution of corrupted inputs."
        ),metadata={"source":"gns.pdf","page":4})
        self.other=[Document(page_content="Physics background "+str(i),
                             metadata={"source":"pinn.pdf","page":i})
                    for i in range(9)]
        self.docs=[self.main]+self.other
    def as_retriever(self,search_kwargs=None):
        return FakeRetriever(self.docs[:4])
    def similarity_search(self,q,k=4,**kwargs):
        return self.docs[:k]
    def get(self,include=None):
        return {"metadatas":[d.metadata for d in self.docs]}


class Chat:
    def __call__(self,inputs):
        return "Training noise protects rollouts [SOURCE gns:pdfpage5]"


def test_unverified_quote_is_retried_and_accepted_after_exact_copy():
    store=FakeStore()
    registry=PaperRegistry([{"short_id":"gns","file":"gns.pdf"},
                            {"short_id":"orig_pinn","file":"pinn.pdf"}])
    rounds=[]
    def grader(question,docs):
        rounds.append([d.page_content for d in docs])
        return [{
            "index":i,"grade":"correct" if d.metadata["source"]=="gns.pdf" else "incorrect",
            "quote":("training noise paraphrased by grader" if len(rounds)==1
                     else d.page_content[:32]) if d.metadata["source"]=="gns.pdf" else "",
            "reason":"relevant" if d.metadata["source"]=="gns.pdf" else "unrelated"
        } for i,d in enumerate(docs)]
    crag=CRAGStrategy(ProvenanceStore(store,registry),Chat(),registry=registry,
                      max_grades=2,grader_fn=grader,pool_k=10)
    out=crag.run_with_trace("Why did the authors inject random-walk noise during training?")
    assert out.trace["crag"]["action"]=="corrected"
    assert len(rounds)==2
    assert out.trace["crag"]["retrying_unverified_quotes"]==1
    assert out.trace["crag"]["correction_retrieval"]["selection_limit"]==8
    assert any(g["quote_verified"] for g in out.trace["crag"]["corrective_grades"])
    assert out.sources[0]["paper_id"]=="gns"


def test_corrective_candidates_remain_bounded_and_diverse():
    store=FakeStore()
    registry=PaperRegistry([{"short_id":"gns","file":"gns.pdf"},
                            {"short_id":"orig_pinn","file":"pinn.pdf"}])
    aware=CRAGStrategy(ProvenanceStore(store,registry),Chat(),registry=registry,
                       grader_fn=lambda q,d:[],pool_k=28)
    docs,trace=aware.source_retriever.retrieve(
        "Explain scientific training",initial=[],widen=True)
    assert len(docs)==8
    assert trace["correction_widened"] and trace["selection_limit"]==8
