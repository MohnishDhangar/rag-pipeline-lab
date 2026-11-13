"""Use an isolated evaluation JSON without falling back to the dev 63 questions."""
import json
from langchain_core.documents import Document
from rag_lab.evaluation import generate, retrieval_matrix
from rag_lab.llms.gpt_adapter import build_chat_model


class FakeStore:
    def __init__(self):
        self.docs=[Document(page_content="The XPINN subdomains enforce continuity.",
                            metadata={"source":"xpinn.pdf","page":1})]
    def as_retriever(self,search_kwargs=None):
        class Ret:
            def __init__(self,docs):self.docs=docs
            def invoke(self,question,config=None,**kwargs):return self.docs
        return Ret(self.docs)


class Transport:
    calls=0
    def generate(self,input,instructions=None):
        self.calls+=1
        return "Interfaces enforce continuity [SOURCE xpinn:pdfpage2]",{}


def test_both_generators_use_selected_holdout_file(tmp_path,monkeypatch):
    (tmp_path/"eval").mkdir()
    development={"metadata":{"papers":[{"short_id":"xpinn","file":"xpinn.pdf"}]},
                 "questions":[{"id":"dev_only","question":"Different dev question?",
                               "source_documents":["xpinn"]}]}
    holdout={"metadata":development["metadata"],
             "questions":[{"id":"holdout_one","question":"How do XPINN domain interfaces work?",
                           "source_documents":["xpinn"],"question_type":"single_document",
                           "ground_truth":"They enforce continuity of nearby solutions."}]}
    (tmp_path/"eval"/"eval_set.json").write_text(json.dumps(development))
    candidate=tmp_path/"eval"/"holdout.json"
    candidate.write_text(json.dumps(holdout))
    monkeypatch.setattr(retrieval_matrix,"create_backend",
                        lambda *args,**kwargs:FakeStore())
    output=tmp_path/"retrieval.jsonl"
    assert retrieval_matrix.benchmark(
        tmp_path,output,backends=("dense",),k=4,
        eval_set=candidate)==1
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert {r["id"] for r in rows}=={"holdout_one"}
    fake_llm=build_chat_model(model="mock",transport=Transport())
    monkeypatch.setattr(generate,"build_chat_model",lambda model=None:fake_llm)
    monkeypatch.setattr(generate,"create_backend",lambda *args,**kwargs:FakeStore())
    gen_path=tmp_path/"generated.jsonl"
    assert generate.generate(
        tmp_path,gen_path,["naive"],["dense"],
        model="mock",eval_set=candidate)==1
    gen_rows=[json.loads(line) for line in gen_path.read_text().splitlines()]
    assert {r["id"] for r in gen_rows}=={"holdout_one"}
