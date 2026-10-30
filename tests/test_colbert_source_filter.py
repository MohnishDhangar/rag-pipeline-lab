"""PyLate subset filtering must recover named papers below global top-k.

The mocks emulate documented retrieve(..., subset=[chunk IDs]) behavior.
No Gemini/GPT traffic or real PLAID index is needed for this contract test.
"""
import pytest
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.colbert_backend import ColBERTVectorStoreShim
from rag_lab.retrieval.provenance import ProvenanceStore
from rag_lab.retrieval.source_aware import SourceAwareRetriever


class QueryModel:
    def encode(self,queries,**kwargs):
        assert kwargs["is_query"]
        return [[1.0]]

class SubsetRetriever:
    def __init__(self):
        self.calls=[]
        self.all_ids=["chunk_0","chunk_1","chunk_2","chunk_3","chunk_4","chunk_5"]
    def retrieve(self,queries_embeddings,k,subset=None):
        self.calls.append({"k":k,"subset":subset})
        matched=self.all_ids if subset is None else [
            value for value in self.all_ids if value in subset
        ]
        return [[{"id":value,"score":20.0-i}
                 for i,value in enumerate(matched[:k])]]


def make_store():
    texts={f"chunk_{i}":f"Scientific paper passage {i}"
           for i in range(6)}
    metas={f"chunk_{i}":{"source":"ntk.pdf" if i<5 else "orig.pdf",
                          "page":i}
           for i in range(6)}
    retriever=SubsetRetriever()
    store=ColBERTVectorStoreShim("unused",texts,metas,
        model=QueryModel(),index=object(),retriever=retriever)
    return store,retriever


def test_exact_source_filter_calls_pylate_subset_and_recovers_rare_paper():
    backend,search=make_store()
    global_docs=backend.similarity_search("PINN training",k=4)
    assert {d.metadata["source"] for d in global_docs}=={"ntk.pdf"}
    result=backend.similarity_search("PINN training",k=4,
                                     filter={"source":"orig.pdf"})
    assert len(result)==1
    assert result[0].metadata["original_id"]=="chunk_5"
    assert search.calls[-1]["subset"]==["chunk_5"]
    assert backend.similarity_search("question",filter={"source":"unknown.pdf"})==[]
    with pytest.raises(ValueError,match="exact source-path"):
        backend.similarity_search("question",filter={"paper_id":"orig_pinn"})


def test_source_aware_colbert_recovers_requested_papers_not_in_global_top28():
    backend,search=make_store()
    registry=PaperRegistry([{"short_id":"ntk_pinn","file":"ntk.pdf"},
                            {"short_id":"orig_pinn","file":"orig.pdf"}])
    store=ProvenanceStore(backend,registry)
    aware=SourceAwareRetriever(store,registry,k=4,pool_k=28)
    initial=store.as_retriever(search_kwargs={"k":4}).invoke("Both NTK-PINN and Raissi et al.")
    docs,trace=aware.retrieve("Both NTK-PINN and Raissi et al. describe PINN training.",
                              initial=initial)
    assert {d.metadata["paper_id"] for d in docs}=={"ntk_pinn","orig_pinn"}
    assert trace["requested_papers"]==["ntk_pinn","orig_pinn"]
    assert not trace["missing_requested_papers"]
    assert any(call["subset"]==["chunk_5"] for call in search.calls)
