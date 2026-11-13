"""Deterministic, zero-model-cost query-type routing.

No ground-truth question_type, source_documents, answer or retrieved paper
metadata may be passed to classify_question. This is an exploratory heuristic
(not a trained or externally validated classifier).
"""
from __future__ import annotations
import re
from dataclasses import dataclass

VERSION = "text-router-v1"


@dataclass(frozen=True)
class RouteDecision:
    question_type: str
    search_type: str
    rule: str
    version: str = VERSION


def classify_question(question: str) -> RouteDecision:
    """Route using ONLY the question text, never evaluation labels."""
    text = " ".join(str(question).split())
    low = text.lower()
    rule = None
    if re.search(r"\b(?:all three papers|three of the papers|these papers)\b", low):
        rule = "explicit_multi_paper"
    elif re.search(r"^\s*both\b", low):
        rule = "both_subjects"
    elif re.search(r"\bpaper\s+and\s+(?:the\s+)?[^.?!]{0,100}\bpaper\s+both\b", low):
        rule = "two_named_papers"
    elif re.search(r"\bwhile\b", low) and re.search(r"\bgns\b", low) and re.search(r"\bpinns?\b", low):
        rule = "contrasting_named_methods"
    elif re.search(r"\bcontrast\b[^.?!]{0,100}\beach\b", low) and low.count("paper") >= 2:
        rule = "contrast_two_papers"
    if rule:
        return RouteDecision("cross_document", "mmr", rule)
    return RouteDecision("single_document", "similarity", "default_conservative")


def routing_search_kwargs(decision: RouteDecision, k=4, fetch_k=20, lambda_mult=.5):
    kwargs = {"k": k}
    if decision.search_type == "mmr":
        kwargs.update(fetch_k=fetch_k, lambda_mult=lambda_mult)
    return kwargs
