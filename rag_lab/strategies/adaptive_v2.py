"""Single-shot router: choose a backend, optionally run corpus-corrective CRAG."""
from rag_lab.core.trace import TraceableStrategy
from rag_lab.routing.controller_v2 import decide,VERSION
from rag_lab.strategies.naive import NaiveStrategy
from rag_lab.strategies.crag import CRAGStrategy


class AdaptiveV2Strategy(TraceableStrategy):
    def __init__(self, stores, llm, registry, *, max_grader_calls=1,
                 crag_pool_k=28, source_query_budget=3):
        if "dense" not in stores: raise ValueError("Adaptive v2 requires dense baseline")
        self.stores,self.llm,self.registry=stores,llm,registry
        self.max_grader_calls=max_grader_calls
        self.crag_pool_k=crag_pool_k
        self.source_query_budget=source_query_budget

    def run_with_trace(self,question):
        decision=decide(question,colbert_available="colbert" in self.stores,
                        max_grader_calls=self.max_grader_calls)
        store=self.stores[decision.backend]
        if decision.strategy=="crag":
            delegate=CRAGStrategy(store,self.llm,registry=self.registry,
                                  max_grades=decision.expected_max_grader_calls,
                                  pool_k=self.crag_pool_k,
                                  max_target_queries=self.source_query_budget)
        else:
            delegate=NaiveStrategy(store,self.llm)
        output=delegate.run_with_trace(question)
        output.strategy=type(self).__name__
        output.trace["controller_v2"]={
            "version":VERSION,"backend":decision.backend,
            "strategy":decision.strategy,"rationale":decision.rationale,
            "max_grader_calls":decision.expected_max_grader_calls,
            "colbert_available":"colbert" in self.stores,
        }
        return output

    def run(self,question):
        return self.run_with_trace(question).answer
