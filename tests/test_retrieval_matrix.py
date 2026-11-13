import json
from langchain_core.documents import Document
from rag_lab.evaluation import retrieval_matrix


class Retriever:
    def invoke(self,question,config=None):
        return [Document(page_content="evidence",metadata={
            "source":"paper.pdf","page":0,"id":"doc0"})]


class FakeBackend:
    def as_retriever(self,search_kwargs=None):
        return Retriever()


def test_backend_retrieval_matrix_offline(tmp_path,monkeypatch):
    root=tmp_path
    (root/"eval").mkdir()
    (root/"eval"/"eval_set.json").write_text(json.dumps({
        "metadata":{"papers":[{"short_id":"paper","file":"paper.pdf"}]},
        "questions":[{"id":"q","question":"Q","source_documents":["paper"]}]}))
    monkeypatch.setattr(retrieval_matrix,"create_backend",
                        lambda name,*args,**kwargs:FakeBackend())
    dest=root/"retrieval.jsonl"
    n=retrieval_matrix.benchmark(root,dest,backends=("dense","colbert"))
    assert n==2
    assert retrieval_matrix.benchmark(root,dest,backends=("dense","colbert"))==0
    rows=[json.loads(line) for line in dest.read_text().splitlines()]
    assert {r["backend"] for r in rows}=={"dense","colbert"}
    assert all(r["paper_recall_at_k"]==1 for r in rows)
