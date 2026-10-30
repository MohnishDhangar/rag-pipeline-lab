from langchain_core.documents import Document
from rag_lab.retrieval.source_aware import requested_sources,SourceAwareRetriever
from rag_lab.knowledge.paper_registry import PaperRegistry


def reg():
    return PaperRegistry([{"short_id":p,"title":p} for p in ("xpinn","sk_pinn","gns","ntk_pinn")])


def doc(p):
    return Document(page_content=p+" evidence",metadata={"source":p+".pdf","page":1})


class FakeStore:
    def __init__(self):
        self.calls=[]
    def similarity_search(self,query,k=4,**kwargs):
        self.calls.append((query,k,kwargs))
        if "xpinn" in query.lower():
            return [doc("xpinn"),doc("sk_pinn")]
        return [doc("sk_pinn")]


def test_no_gold_dependency_or_generic_pinn_inference():
    assert requested_sources("Both XPINN and SK-PINN improve PINNs.").paper_ids==("xpinn","sk_pinn")
    q=requested_sources("All three papers in this batch approach the task differently.")
    assert q.unresolved_scope and not q.paper_ids
    assert requested_sources("What are PINNs?").paper_ids==()


def test_targeted_retrieval_recovers_missing_source_without_fake_labels():
    registry=PaperRegistry([
        {"short_id":"xpinn","title":"xpinn","file":"xpinn.pdf"},
        {"short_id":"sk_pinn","title":"sk_pinn","file":"sk_pinn.pdf"}])
    store=FakeStore()
    retriever=SourceAwareRetriever(store,registry,k=4)
    # No indexed file map, so this fake store uses global candidate searches.
    docs,trace=retriever.retrieve("Both XPINN and SK-PINN compare approaches.")
    assert set(registry.resolve(d.metadata["source"]) for d in docs)=={"xpinn","sk_pinn"}
    assert not trace["missing_requested_papers"]
