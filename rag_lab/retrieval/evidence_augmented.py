"""Blend existing dense/ColBERT retrieval with local paper evidence graph; zero GPT calls."""
from __future__ import annotations
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

class EvidenceAugmentedVectorStore:
    def __init__(self, base, graph):
        self.base, self.graph = base, graph

    def similarity_search(self, query, k=4):
        # RRF operates on ranking, not incomparable dense/late-interaction scores.
        base_docs=self.base.similarity_search(query,k=k)
        graph_docs=[
            Document(page_content=e.text,
                     metadata={"id":e.id,"paper_id":e.paper_id,"page":e.page,
                               "source":e.source,"retrieval_type":"evidence_graph"})
            for e in self.graph.query(query,k=k)
        ]
        scores, seen = {}, {}
        for weight,docs in [(1.0,base_docs),(0.6,graph_docs)]:
            for rank,doc in enumerate(docs):
                meta=doc.metadata or {}
                key=str(meta.get("id") or meta.get("source") or doc.page_content)
                scores[key]=scores.get(key,0.0)+weight/(60+rank+1)
                seen[key]=doc
        return [seen[x] for x in sorted(scores,key=scores.get,reverse=True)[:k]]

    def as_retriever(self, search_kwargs=None):
        return _EvidenceRetriever(self,(search_kwargs or {}).get("k",4))

class _EvidenceRetriever(Runnable):
    def __init__(self,store,k): self.store,self.k=store,k
    def invoke(self,input,config=None,**kwargs):
        return self.store.similarity_search(input,k=self.k)
