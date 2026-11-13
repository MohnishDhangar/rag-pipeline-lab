"""Cost-capped, query-only routing between dense/ColBERT and ordinary/CRAG.

Experimental zero-model-cost policy; frozen independently of benchmark gold.
May route to a less expensive safe fallback when ColBERT index unavailable.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from rag_lab.retrieval.query_router import classify_question
from rag_lab.retrieval.source_aware import requested_sources

VERSION="controller-v2.0"
TECHNICAL=re.compile(
    r"\b(?:PDE|residual|Laplacian|derivative|gradient|equation|"
    r"boundary|finite.difference|numerical|random.walk|noise|"
    r"pressure|accuracy|error|inverse problem)\b",re.I)


@dataclass(frozen=True)
class PolicyDecision:
    backend: str
    strategy: str
    rationale: str
    expected_max_grader_calls: int
    version: str=VERSION


def decide(question, *, colbert_available=False, max_grader_calls=1):
    if max_grader_calls not in (0,1,2):
        raise ValueError("grader budget must be 0, 1, or 2")
    query_route=classify_question(question)
    named=requested_sources(question)
    technical=bool(TECHNICAL.search(question))
    if query_route.question_type=="cross_document":
        backend="colbert" if colbert_available else "dense"
        if max_grader_calls:
            return PolicyDecision(backend,"crag","comparative_question_requires_checked_evidence",
                                  max_grader_calls)
        return PolicyDecision(backend,"naive","grader_budget_exhausted",0)
    if named.paper_ids and technical and max_grader_calls:
        return PolicyDecision("colbert" if colbert_available else "dense",
                              "crag","named_scientific_source_with_technical_claims",
                              max_grader_calls)
    if technical and colbert_available:
        return PolicyDecision("colbert","naive","technical_late_interaction",0)
    return PolicyDecision("dense","naive","cheap_dense_default",0)
