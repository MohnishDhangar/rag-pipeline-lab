"""Regression tests for intentional abstention and CRAG reretrieval."""
import json
from langchain_core.documents import Document
from rag_lab.evaluation import generate
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.source_aware import SourceAwareRetriever
from rag_lab.strategies.crag import ABSTAIN


class Retriever:
    def __init__(self,docs):
        self.docs=docs
    def invoke(self,question,config=None,**kwargs):
        return self.docs


class FakeStore:
    def __init__(self,docs):
        self.docs=docs
    def as_retriever(self,search_kwargs=None):
        return Retriever(self.docs)
    def similarity_search(self,query,k=4,**kwargs):
        return self.docs[:k]


class FakeModel:
    model_name="mock"
    def __init__(self):
        self.calls=0


def test_crag_abstention_does_not_truncate_four_way_benchmark(tmp_path,monkeypatch):
    root=tmp_path
    (root/"eval").mkdir()
    (root/"eval"/"eval_set.json").write_text(json.dumps({
        "metadata":{"papers":[{"short_id":"xpinn","file":"xpinn.pdf"}]},
        "questions":[{
            "id":"scope","question":"Compare all three papers in this batch.",
            "source_documents":["xpinn"],"ground_truth":"Test",
            "question_type":"cross_document"
        }]
    }))
    docs=[Document(page_content="XPINN scientific evidence",
                   metadata={"source":"xpinn.pdf","page":0})]
    # Use an actual trace model, but the CRAG scope branch must make no
    # generation/grading calls when the batch is unspecified.
    from rag_lab.llms.gpt_adapter import build_chat_model
    class Transport:
        calls=0
        def generate(self,input,instructions=None):
            self.calls+=1
            return "Supported [SOURCE xpinn:pdfpage1]",{}
    transport=Transport()
    model=build_chat_model(model="mock",transport=transport)
    monkeypatch.setattr(generate,"create_backend",
                        lambda *args,**kwargs:FakeStore(docs))
    monkeypatch.setattr(generate,"build_chat_model",lambda model=None:model_obj)
    model_obj=model
    target=root/"generated.jsonl"
    assert generate.generate(root,target,["crag"],["dense","colbert"],model="mock")==2
    rows=[json.loads(x) for x in target.read_text().splitlines()]
    assert len(rows)==2
    assert all(r["answer"]==ABSTAIN and r["generation_status"]=="abstained" for r in rows)
    assert transport.calls==0
    assert generate.generate(root,target,["crag"],["dense","colbert"],model="mock")==0


def test_correction_excludes_rejected_top_k():
    registry=PaperRegistry([{"short_id":"xpinn","file":"xpinn.pdf"}])
    docs=[Document(page_content=f"Passage {i}",
                   metadata={"source":"xpinn.pdf","page":i}) for i in range(9)]
    store=FakeStore(docs)
    retriever=SourceAwareRetriever(store,registry,k=4,pool_k=9)
    before=docs[:4]
    after,info=retriever.retrieve(
        "Which passage contains the evidence?",requested=None,initial=[],
        exclude_keys={(d.metadata["source"],d.metadata["page"],d.page_content)
                      for d in before},widen=True)
    assert [d.page_content for d in after] == [f"Passage {i}" for i in range(4,9)]
    assert len(after)<=8
    assert info["correction_widened"]
    assert info["excluded_previous_passages"]==4
