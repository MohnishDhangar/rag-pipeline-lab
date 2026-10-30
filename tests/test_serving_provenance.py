"""Serving must use the same canonical source metadata as offline evaluation."""
import json
from langchain_core.documents import Document
from rag_lab.application.serving import serving_store


class Dummy:
    def similarity_search(self, query, k=4, **kwargs):
        return [Document(page_content="A cited scientific fact.",
                         metadata={"source":"/papers/gns.pdf","page":0})]


def test_serving_provenance_does_not_need_gold_question_labels(tmp_path):
    path=tmp_path/"registry.json"
    path.write_text(json.dumps({"metadata":{"papers":[
        {"short_id":"gns","file":"gns.pdf"}]},"questions":[]}))
    store,registry=serving_store(Dummy(),registry_path=path)
    docs=store.similarity_search("unseen scientific question")
    assert registry.resolve("gns.pdf")=="gns"
    assert docs[0].metadata["paper_id"]=="gns"
    assert docs[0].metadata["pdf_page_number"]==1
    assert docs[0].metadata["id"].startswith("gns:")
