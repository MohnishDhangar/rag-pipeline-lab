"""Exactly one GPT call on evidence selected by the text-only retrieval router."""
from rag_lab.strategies.naive import NaiveStrategy


class AdaptiveNaiveStrategy(NaiveStrategy):
    def run_with_trace(self, question):
        result = super().run_with_trace(question)
        raw = getattr(self.vectorstore, "base", self.vectorstore)
        result.trace["retrieval_route"] = getattr(raw, "last_route", None)
        return result
