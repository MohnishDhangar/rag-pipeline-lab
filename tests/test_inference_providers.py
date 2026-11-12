import json
import pytest
import requests
from pydantic import BaseModel
from rag_lab.llms.factory import build_chat_model, resolve_provider, inference_identity
from rag_lab.llms.providers import APITransport
from rag_lab.llms.gpt_adapter import InferenceError

class Grade(BaseModel):
    grade: str

class Response:
    def __init__(self, payload, status=200):
        self.payload=payload
        self.status_code=status
        self.ok=status==200
        self.headers={}
    def json(self):
        return self.payload

class Session:
    def __init__(self,response):
        self.response=response
        self.sent=[]
    def post(self,url,**kwargs):
        self.sent.append((url,kwargs))
        return self.response

def groq(text, finish="stop"):
    return Response({"choices":[{"finish_reason":finish,"message":{"content":text}}],
                     "usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9}})

def test_groq_text_and_native_schema(monkeypatch):
    monkeypatch.delenv("RAG_LLM_CACHE",raising=False)
    session=Session(groq('{"grade":"correct"}'))
    t=APITransport("groq","openai/gpt-oss-120b",http=session,api_key="fake")
    llm=build_chat_model(provider="groq",model=t.model,transport=t)
    assert llm.invoke("question").content=='{"grade":"correct"}'
    assert llm.with_structured_output(Grade).invoke("question").grade=="correct"
    assert t.calls==2 and t.usage_totals["total_tokens"]==18
    payload=session.sent[-1][1]["json"]
    assert payload["stream"] is False
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["grade"]["type"]=="string"
    assert "fake" not in json.dumps(payload)

def test_gemini_request_roles_and_usage(monkeypatch):
    monkeypatch.delenv("RAG_LLM_CACHE",raising=False)
    session=Session(Response({"candidates":[{"finishReason":"STOP","content":{"parts":[
        {"text":"private reasoning","thought":True},{"text":'{"grade":"correct"}'}]}}],
        "usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"thoughtsTokenCount":2,"totalTokenCount":9}}))
    t=APITransport("gemini","models/test-model",http=session,api_key="fake")
    text,usage=t.generate_structured([{"role":"user","content":"q"}],"system",Grade)
    assert json.loads(text)=={"grade":"correct"}
    assert usage["output_tokens"]==6
    url,req=session.sent[0]
    assert url.endswith("/models/test-model:generateContent")
    assert req["json"]["systemInstruction"]["parts"][0]["text"]=="system"
    assert req["json"]["generationConfig"]["responseMimeType"]=="application/json"
    assert "fake" not in json.dumps(req["json"])

@pytest.mark.parametrize("response,code",[(Response({},429),"rate_limit"),(groq("truncated","length"),"incomplete_response")])
def test_provider_failure(response,code,monkeypatch):
    monkeypatch.delenv("RAG_LLM_CACHE",raising=False)
    t=APITransport("groq","test",http=Session(response),api_key="secret")
    with pytest.raises(InferenceError) as err:
        t.generate([{"role":"user","content":"q"}])
    assert err.value.code==code
    assert "secret" not in str(err.value)
    assert t.calls==1

def test_schema_validation_failure(monkeypatch):
    monkeypatch.delenv("RAG_LLM_CACHE",raising=False)
    t=APITransport("groq","test",http=Session(groq('{"wrong":1}')),api_key="fake")
    model=build_chat_model(provider="groq",model="test",transport=t)
    with pytest.raises(InferenceError,match="schema validation"):
        model.with_structured_output(Grade).invoke("q")

def test_cache_namespaces_and_identity(tmp_path,monkeypatch):
    monkeypatch.setenv("RAG_LLM_CACHE",str(tmp_path/"cache.sqlite"))
    t=APITransport("groq","test",http=Session(groq("answer")),api_key="fake")
    items=[{"role":"user","content":"q"}]
    assert t.generate(items)[0]=="answer"
    assert t.generate(items)[1]["cached"]
    assert t.calls==1
    assert inference_identity("groq","same")!=inference_identity("gemini","same")
