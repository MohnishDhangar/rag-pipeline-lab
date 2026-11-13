"""End-to-end mock of the four-arm generation pipeline, without credentials."""
import json
from langchain_core.documents import Document
from rag_lab.llms.gpt_adapter import build_chat_model
from rag_lab.evaluation import generate


class FakeTransport:
    def __init__(self):
        self.calls=0
    def generate(self,input,instructions=None):
        self.calls+=1
        if instructions and "Return ONLY JSON matching this schema" in instructions:
            response={"assessments":[
                {"index":0,"grade":"correct",
                 "quote":"XPINN uses domain decomposition","reason":"supports question"},
                {"index":1,"grade":"correct",
                 "quote":"SK-PINN uses smoothing gradients","reason":"supports question"},
            ]}
            return json.dumps(response),{"input_tokens":20,"output_tokens":10,"total_tokens":30}
        return "XPINN and SK-PINN differ in training.", {
            "input_tokens":30,"output_tokens":15,"total_tokens":45
        }


class FakeRetriever:
    def __init__(self,docs):
        self.docs=docs
    def invoke(self,question,config=None,**kwargs):
        return self.docs


class FakeBackend:
    def __init__(self):
        self.docs=[
            Document(page_content="XPINN uses domain decomposition",
                     metadata={"source":"xpinn.pdf","page":1}),
            Document(page_content="SK-PINN uses smoothing gradients",
                     metadata={"source":"sk_pinn.pdf","page":2}),
        ]
    def as_retriever(self,search_kwargs=None):
        return FakeRetriever(self.docs)
    def similarity_search(self,query,k=4,**kwargs):
        return self.docs[:k]


def test_generate_four_matrix_with_correct_contexts_and_traces(tmp_path,monkeypatch):
    root=tmp_path
    (root/"eval").mkdir()
    (root/"eval"/"eval_set.json").write_text(json.dumps({
        "metadata":{"papers":[{"short_id":"xpinn","file":"xpinn.pdf"},
                              {"short_id":"sk_pinn","file":"sk_pinn.pdf"}]},
        "questions":[{
            "id":"q","question":"Both XPINN and SK-PINN improve scientific training.",
            "question_type":"cross_document","source_documents":["xpinn","sk_pinn"],
            "ground_truth":"Two distinct methods"
        }]
    }))
    monkeypatch.setattr(generate,"create_backend",lambda *args,**kwargs:FakeBackend())
    transport=FakeTransport()
    model=build_chat_model(model="mock",transport=transport)
    monkeypatch.setattr(generate,"build_chat_model",lambda model=None:model_obj)
    model_obj=model
    output=root/"gen.jsonl"
    done=generate.generate(root,output,["naive","crag"],["dense","colbert"],
                           model="mock",crag_max_grades=1)
    assert done==4
    rows=[json.loads(x) for x in output.read_text().splitlines()]
    assert len(rows)==4
    assert {(r["backend"],r["strategy"]) for r in rows} == {
        ("dense","naive"),("dense","crag"),
        ("colbert","naive"),("colbert","crag")}
    assert all(len(r["sources"])==2 for r in rows)
    assert all({s["paper_id"] for s in r["sources"]}=={"xpinn","sk_pinn"} for r in rows)
    assert all(r["trace"]["gpt_requests"]==2 for r in rows if r["strategy"]=="crag")
    assert all(r["trace"]["gpt_requests"]==1 for r in rows if r["strategy"]=="naive")
    # Checkpoint must not send another GPT query.
    assert generate.generate(root,output,["naive","crag"],["dense","colbert"],
                             model="mock",crag_max_grades=1)==0
    assert transport.calls==6
