"""
ColBERTVectorStoreShim mimics just the two interface points every
existing strategy actually calls on a Chroma vectorstore:
  - .as_retriever(search_kwargs={"k": k}) -> object with .invoke(query)
    used by: naive, multi_query, rag_fusion, step_back, decomposition
  - .similarity_search(query, k=k) -> List[Document]
    used by: hyde (calls this directly, no as_retriever())

This lets ColBERT plug into every strategy file UNCHANGED, rather than
refactoring all six to accept an injected retriever — avoiding any risk
to strategies already confirmed working against dense retrieval.
Interface-matching validated against mocks of each call pattern before
this was written into the strategies.
"""

from typing import List
from langchain_core.documents import Document
from ragatouille import RAG


class _ColBERTRetriever:
    def __init__(self, rag: RAG, k: int):
        self.rag = rag
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        results = self.rag.search(query, k=self.k)
        return [
            Document(page_content=r["content"], metadata={"score": r["score"], "rank": r["rank"]})
            for r in results
        ]


class ColBERTVectorStoreShim:
    def __init__(self, index_path: str):
        self.rag = RAG.from_index(index_path)

    def as_retriever(self, search_kwargs: dict = None):
        k = (search_kwargs or {}).get("k", 4)
        return _ColBERTRetriever(self.rag, k)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        results = self.rag.search(query, k=k)
        return [
            Document(page_content=r["content"], metadata={"score": r["score"], "rank": r["rank"]})
            for r in results
        ]