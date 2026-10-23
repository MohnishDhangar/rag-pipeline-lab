import json
import pytest
from rag_lab.llms.chatgpt_oauth import CredentialStore
from rag_lab.llms.gpt_adapter import GPTTransport, InferenceError
from rag_lab.evaluation.judge import evaluate_row

class Auth:
    def token(self): return "fake-token"

class Stream:
    ok=True
    status_code=200
    def __init__(self,events): self.events=events
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def iter_lines(self,decode_unicode=False):
        for item in self.events:
            yield "data: "+json.dumps(item)

class Session:
    def __init__(self,events):self.events=events;self.payload=None
    def post(self,url,headers,json,stream,timeout):
        self.payload=json
        return Stream(self.events)

def test_stream_must_finish_and_disallows_unsupported_fields():
    events=[{"type":"response.output_text.delta","delta":"Hello"},
            {"type":"response.completed","response":{"usage":{"input_tokens":4,"output_tokens":1,"total_tokens":5}}}]
    fake=Session(events)
    result=GPTTransport(auth=Auth(),http=fake,model="test-slug")
    assert result.generate([{"role":"user","content":"hi"}])[0]=="Hello"
    assert fake.payload["stream"] is True and fake.payload["store"] is False
    assert "temperature" not in fake.payload
    with pytest.raises(InferenceError,match="response.completed"):
        GPTTransport(auth=Auth(),http=Session(events[:-1]),model="x").generate([{"role":"user","content":"hi"}])

def test_credential_storage_does_not_expose_secrets(tmp_path):
    store=CredentialStore(tmp_path)
    assert store.host_id()==store.host_id()
    store.save({"access_token":"secret","client_id":"example"})
    assert store.load()["access_token"]=="secret"
    assert not (tmp_path/"session.json").is_symlink()

def test_single_call_judge():
    class Fake:
        model="test-judge"
        def generate(self,input,instructions=None):
            return json.dumps(dict(faithfulness=1,answer_correctness=.8,
                                  answer_relevance=.9,citation_support=0,reason="No citations")),{}
    row=dict(question="q",answer="a",ground_truth="x",contexts=["context"])
    result=evaluate_row(row,Fake())
    assert result["judge_version"]=="scientific-judge-v1"
    assert result["citation_support"]==0
