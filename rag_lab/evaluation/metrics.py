"""Deterministic paper-level retrieval metrics (no judge model required)."""
from __future__ import annotations
import math

def score_retrieval(retrieved_paper_ids, relevant_paper_ids, k=4):
    """Precision/recall/MRR/nDCG@k using annotated source paper IDs.

    Unknown source metadata is never treated as ground-truth relevant.
    Multiple chunks from one paper count once for paper-level evaluation.
    """
    relevant = set(relevant_paper_ids or [])
    ranked = list(dict.fromkeys(p for p in retrieved_paper_ids if p))[:k]
    if not relevant:
        return {"precision_at_k":None,"recall_at_k":None,
                "mrr_at_k":None,"ndcg_at_k":None}
    hits = [int(p in relevant) for p in ranked]
    dcg = sum(h/math.log2(i+2) for i,h in enumerate(hits))
    ideal = sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))
    return {"precision_at_k":sum(hits)/k,
            "recall_at_k":len(set(ranked)&relevant)/len(relevant),
            "mrr_at_k":next((1/(i+1) for i,h in enumerate(hits) if h),0.0),
            "ndcg_at_k":dcg/ideal if ideal else 0.0}
