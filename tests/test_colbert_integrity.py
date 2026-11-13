import pytest
from rag_lab.retrieval.colbert_backend import ColBERTVectorStoreShim,validate_mappings


class Model:
    def encode(self,queries,**kwargs):
        assert kwargs["is_query"]
        return [[1]]

class Retriever:
    def __init__(self,ids):
        self.ids=ids
    def retrieve(self,queries_embeddings,k):
        return [[{"id":name,"score":12.0-rank} for rank,name in enumerate(self.ids[:k])]]


def test_colbert_mapping_exact_and_no_fake_context():
    with pytest.raises(ValueError,match="metadata/text IDs differ"):
        validate_mappings({"chunk_0":"real text"},{})
    texts={"chunk_0":"real evidence","chunk_1":"second evidence"}
    metadata={k:{"source":"paper.pdf","page":i} for i,k in enumerate(texts)}
    store=ColBERTVectorStoreShim("unused",texts,metadata,model=Model(),index=object(),
                                retriever=Retriever(["chunk_1","chunk_0"]))
    docs=store.as_retriever(search_kwargs={"k":2}).invoke("scientific question")
    assert [d.page_content for d in docs]==["second evidence","real evidence"]
    assert docs[0].metadata["page"]==1
    assert docs[0].metadata["score"]==12.0
    with pytest.raises(ValueError,match="MMR"):
        store.as_retriever(search_type="mmr")


def test_unknown_colbert_ids_fail_closed():
    store=ColBERTVectorStoreShim("unused",{"chunk_0":"real"},
                                {"chunk_0":{"source":"paper.pdf","page":0}},
                                model=Model(),index=object(),retriever=Retriever(["chunk_999"]))
    with pytest.raises(KeyError,match="misaligned"):
        store.similarity_search("scientific question")
