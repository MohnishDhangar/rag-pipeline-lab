"""Cheap evidence sufficiency policy. No LLM invocation for the gate."""
from dataclasses import dataclass
from rag_lab.knowledge.citations import citation_coverage

@dataclass
class ReliabilityVerdict:
    action: str
    reasons: list[str]

def assess(result, *, cross_document=False, require_citations=False):
    reasons=[]
    if not result.contexts:
        return ReliabilityVerdict("abstain",["No retrieved evidence"])
    papers={s.get("paper_id") for s in result.sources if s.get("paper_id")}
    if cross_document and len(papers)<2:
        reasons.append("Insufficient distinct paper provenance for cross-document answer")
    if require_citations:
        check=citation_coverage(result.answer,result.sources)
        if check["validity"] is None or check["validity"]<1.0:
            reasons.append("Citation provenance absent or invalid")
    return ReliabilityVerdict("verify" if reasons else "accept",reasons)
