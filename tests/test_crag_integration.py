"""No network calls: full LangChain structured grading -> CRAG -> RAGResult."""
import json
from langchain_core.documents import Document
from rag_lab.llms.gpt_adapter import build_chat_model
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.provenance import ProvenanceStore
from rag_lab.strategies.crag import CRAGStrategy, ABSTAIN


class DummyRetriever:
    def __init__(self, docs):
        self.docs=docs
    def invoke(self,query,config=None,**kwargs):
        return self.docs


class DummyStore:
    def __init__(self,docs):
        self.docs=docs
    def as_retriever(self,search_kwargs=None):
        return DummyRetriever(self.docs)
    def similarity_search(self,query,k=4,**kwargs):
        return self.docs[:k]


class FakeTransport:
    def __init__(self):
        self.calls=0
        self.model="mock-gpt"
    def generate(self,input,instructions=None):
        self.calls+=1
        if instructions and "JSON matching this schema" in instructions:
            # Query covers the two supplied papers; each quote is verbatim.
            body=json.dumps({"assessments":[
                {"index":0,"grade":"correct","quote":"XPINN uses domain decomposition","reason":"direct"},
                {"index":1,"grade":"correct","quote":"SK-PINN uses kernel gradients","reason":"direct"}]})
            return body,{"input_tokens":10,"output_tokens":8,"total_tokens":18}
        return "The papers differ [SOURCE mock:pdfpage1].", {
            "input_tokens":10,"output_tokens":7,"total_tokens":17
        }


def test_crag_run_with_trace_real_adapter_and_metadata():
    registry=PaperRegistry([
        {"short_id":"xpinn","file":"xpinn.pdf"},
        {"short_id":"sk_pinn","file":"sk_pinn.pdf"}])
    docs=[
        Document(page_content="XPINN uses domain decomposition",
                 metadata={"source":"xpinn.pdf","page":0}),
        Document(page_content="SK-PINN uses kernel gradients",
                 metadata={"source":"sk_pinn.pdf","page":0}),
    ]
    transport=FakeTransport()
    llm=build_chat_model(model="mock-gpt",transport=transport)
    store=ProvenanceStore(DummyStore(docs),registry)
    s=CRAGStrategy(store,llm,registry=registry,max_grades=1)
    result=s.run_with_trace("Both XPINN and SK-PINN differ in how they train.")
    assert len(result.contexts)==2
    assert {src["paper_id"] for src in result.sources}=={"xpinn","sk_pinn"}
    assert result.trace["gpt_requests"]==2
    assert transport.calls==2
    assert result.trace["crag"]["action"]=="correct"


def test_crag_refuses_unidentifiable_papers_without_model_call():
    registry=PaperRegistry([{"short_id":"xpinn","file":"xpinn.pdf"}])
    transport=FakeTransport()
    model=build_chat_model(model="mock-gpt",transport=transport)
    s=CRAGStrategy(DummyStore([]),model,registry=registry)
    result=s.run_with_trace("Compare all three papers in this batch.")
    assert result.answer==ABSTAIN
    assert not result.sources
    assert result.trace["gpt_requests"]==0
    assert transport.calls==0
