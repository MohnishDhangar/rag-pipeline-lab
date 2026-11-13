import json
import pytest
from rag_lab.evaluation.provider_judge import judge_file
from rag_lab.evaluation.identity import row_fingerprint
from rag_lab.llms.gpt_adapter import InferenceError

class Fake:
    provider="groq"
    model="test"
    def __init__(self): self.calls=0
    def generate(self,items,instructions=None):
        self.calls+=1
        return json.dumps(dict(faithfulness=1,answer_correctness=.8,
            answer_relevance=1,citation_support=.9,reason="test")),{}

def test_provider_judge_manifest_and_resume(tmp_path):
    inp=tmp_path/"gen.jsonl"
    row=dict(id="q",strategy="naive",backend="dense",question="q",answer="a",
             contexts=["passage"],sources=[],ground_truth="ref",model="generator",
             inference={"provider":"gemini","model":"generator"})
    abstain={**row,"id":"q2","answer":"Insufficient retrieved evidence to answer this question.",
             "generation_status":"abstained"}
    inp.write_text("\n".join(json.dumps(r) for r in (row,abstain))+"\n")
    out=tmp_path/"judge.jsonl"
    t=Fake()
    assert judge_file(inp,out,provider="groq",model="test",transport=t)==1
    assert judge_file(inp,out,provider="groq",model="test",transport=t)==0
    assert t.calls==1
    result=json.loads(out.read_text())
    assert result["input_fingerprint"]==row_fingerprint(row)
    assert result["judge_inference"]["provider"]=="groq"
    with pytest.raises(ValueError,match="experiment mismatch"):
        judge_file(inp,out,provider="gemini",model="test",transport=t)

def test_provider_judge_interruption_preserves_checkpoint(tmp_path):
    inp=tmp_path/"gen.jsonl"
    row=dict(id="q",strategy="naive",backend="dense",question="q",answer="a",contexts=["e"],sources=[])
    inp.write_text(json.dumps(row)+"\n")
    out=tmp_path/"judge.jsonl"
    class Limited(Fake):
        def generate(self,*a,**kw): raise InferenceError("limited",status=429)
    assert judge_file(inp,out,provider="groq",model="test",transport=Limited())==0
    assert not out.exists()
    assert judge_file(inp,out,provider="groq",model="test",transport=Fake())==1
