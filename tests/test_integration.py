"""Integration tests avoid network credentials and actual LLM usage."""
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rag_lab.llms.gpt_adapter import build_chat_model
from rag_lab.strategies.naive import NaiveStrategy
from rag_lab.core.pipeline import EvidenceAwarePipeline

class FakeTransport:
    def __init__(self):
        self.calls=0
    def generate(self,input,instructions=None):
        self.calls+=1
        if instructions and "JSON matching" in instructions:
            return '{"grade": "correct"}', {"input_tokens":20,"output_tokens":6,"total_tokens":26}
        return "PINNs encode physics [SOURCE paper.pdf:page2]", {"input_tokens":45,"output_tokens":11,"total_tokens":56}

class FakeRetriever:
    def invoke(self,query):
        return [Document(page_content="PINNs encode physical constraints.",metadata={"source":"paper.pdf","page":2})]

class FakeStore:
    def as_retriever(self,search_kwargs=None):
        return FakeRetriever()

def test_langchain_adapter_strategy_and_provenance():
    provider=FakeTransport()
    model=build_chat_model(model="test",transport=provider)
    strategy=NaiveStrategy(FakeStore(),llm=model)
    result=strategy.run_with_trace("Explain PINNs")
    assert result.trace["gpt_requests"]==1
    assert provider.calls==1  # exact retrieved contexts, no second retrieval
    assert result.sources[0]["id"]=="paper.pdf"
    assert result.sources[0]["page"]==2
    assert result.contexts==["PINNs encode physical constraints."]
    pipe=EvidenceAwarePipeline(strategy)
    checked=pipe.run_with_trace("Explain PINNs")
    assert checked.trace["reliability"]["action"]=="accept"

class Grade(BaseModel):
    grade: str

def test_structured_router_uses_json_adapter():
    model=build_chat_model(model="test",transport=FakeTransport())
    prompt=ChatPromptTemplate.from_template("Grade this question: {question}")
    grade=(prompt|model.with_structured_output(Grade)).invoke({"question":"Q"})
    assert grade.grade=="correct"

def test_conditional_verification_only_if_flagged():
    class Dummy:
        def run_with_trace(self,q):
            from rag_lab.core.trace import RAGResult
            return RAGResult(question=q,answer="Unsupported",strategy="naive",
                             contexts=["evidence"],sources=[{"id":"doc"}],latency_ms=1)
    class Verifier:
        def generate(self,input,instructions=None):
            return '{"supported": false, "reason": "No support"}',{}
    result=EvidenceAwarePipeline(Dummy(),verify_on_warning=True,verifier=Verifier()).run_with_trace("q")
    assert "could not verify" in result.answer
    assert result.trace["support_check"]["supported"] is False
