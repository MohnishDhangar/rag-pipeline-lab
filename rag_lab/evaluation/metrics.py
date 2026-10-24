"""Retrieval metrics over *retrieved chunks*, with paper relevance annotations.

No LLM needed. Precision is chunk-level; Recall and MRR use the corresponding
paper IDs; nDCG only rewards the first retrieved chunk from each relevant paper.
Unknown source IDs invalidate ranking metrics rather than being called wrong.
"""
import math


def score_retrieval(retrieved_paper_ids, relevant_paper_ids, k=4):
    if k < 1: raise ValueError("k must be positive")
    ranked=list(retrieved_paper_ids or [])[:k]
    relevant=set(relevant_paper_ids or [])
    coverage=sum(bool(p) for p in ranked)/len(ranked) if ranked else 0.0
    base={"k":k,"retrieved_chunks":len(ranked),
          "provenance_coverage":coverage,
          "precision_at_k":None,"chunk_precision_at_k":None,
          "recall_at_k":None,"paper_recall_at_k":None,
          "mrr_at_k":None,"ndcg_at_k":None,
          "paper_hit_at_k":None}
    if not relevant or not ranked or any(not p for p in ranked):
        return base
    hits=[int(p in relevant) for p in ranked]
    chunk_precision=sum(hits)/k
    seen=set()
    novelty=[]
    for paper in ranked:
        fresh=int(paper in relevant and paper not in seen)
        novelty.append(fresh)
        seen.add(paper)
    recall=len(relevant.intersection(seen))/len(relevant)
    dcg=sum(h/math.log2(i+2) for i,h in enumerate(novelty))
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))
    return {**base, "precision_at_k":chunk_precision,
            "chunk_precision_at_k":chunk_precision,
            "recall_at_k":recall, "paper_recall_at_k":recall,
            "mrr_at_k":next((1/(i+1) for i,h in enumerate(hits) if h),0.0),
            "ndcg_at_k":dcg/ideal if ideal else 0.0,
            "paper_hit_at_k":int(bool(relevant.intersection(seen)))}
