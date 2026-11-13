"""Provenance decorator for Chroma, PyLate, and graph retrieval: no re-embedding."""
from __future__ import annotations
from langchain_core.documents import Document
from langchain_core.runnables import Runnable


class ProvenanceStore:
    def __init__(self, base, registry):
        self.base, self.registry=base,registry

    def _decorate(self, docs):
        return [
            Document(page_content=d.page_content,
                     metadata=self.registry.enrich(d.metadata,d.page_content))
            for d in docs
        ]

    def similarity_search(self, query, k=4, **kwargs):
        return self._decorate(self.base.similarity_search(query,k=k,**kwargs))

    def as_retriever(self, search_kwargs=None):
        return _ProvenanceRetriever(self,
            self.base.as_retriever(search_kwargs=search_kwargs))


class _ProvenanceRetriever(Runnable):
    def __init__(self, owner, underlying):
        self.owner,self.underlying=owner,underlying

    def invoke(self, input, config=None, **kwargs):
        return self.owner._decorate(self.underlying.invoke(input,config=config,**kwargs))
