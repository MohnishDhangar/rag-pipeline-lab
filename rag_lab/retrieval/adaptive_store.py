"""Dense retrieval wrapper selecting similarity or MMR from question text.

The actual query is unchanged; selects exactly one retriever per question.
No gold document IDs, question_type or content is inspected.
"""
from langchain_core.runnables import Runnable

from rag_lab.retrieval.query_router import classify_question, routing_search_kwargs


class AdaptiveRetrievalStore:
    def __init__(self, base, *, k=4, fetch_k=20, lambda_mult=.5):
        if not 0 <= lambda_mult <= 1:
            raise ValueError("MMR lambda_mult must be between 0 and 1")
        if fetch_k < k:
            raise ValueError("fetch_k must be >= k")
        self.base = base
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.last_route = None

    def as_retriever(self, search_kwargs=None):
        k = (search_kwargs or {}).get("k", self.k)
        return _AdaptiveRetriever(self, k)

    def similarity_search(self, query, k=4, **kwargs):
        return self.base.similarity_search(query, k=k, **kwargs)


class _AdaptiveRetriever(Runnable):
    def __init__(self, parent, k):
        self.parent, self.k = parent, k

    def invoke(self, input, config=None, **kwargs):
        decision = classify_question(input)
        parameters = routing_search_kwargs(
            decision, k=self.k,
            fetch_k=max(self.k, self.parent.fetch_k),
            lambda_mult=self.parent.lambda_mult,
        )
        retriever = self.parent.base.as_retriever(
            search_type=decision.search_type, search_kwargs=parameters)
        docs = retriever.invoke(input, config=config, **kwargs)
        self.parent.last_route = {
            "question_type_prediction": decision.question_type,
            "mode": decision.search_type,
            "rule": decision.rule,
            "router_version": decision.version,
            "search_kwargs": parameters,
        }
        return docs
