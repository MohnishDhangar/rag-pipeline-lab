"""Cheap, non-generative query features for adaptive routing."""
import re

FEATURE_NAMES=["tokens","characters","question_marks","comma_count","and_count",
               "comparison","cross_paper","complexity","specificity","math","open_ended"]

def features(question):
    q=question.lower()
    tokens=re.findall(r"[a-z0-9]+",q)
    return [
        len(tokens),
        len(question),
        question.count("?"),
        question.count(","),
        len(re.findall(r"\band\b",q)),
        int(bool(re.search(r"\b(compare|contrast|versus|vs|difference|similar)\b",q))),
        int(bool(re.search(r"\b(both|papers|studies|authors|across|et al)\b",q))),
        int(bool(re.search(r"\b(how does|why does|relate|trade.?off|synthesiz|combine)\b",q))),
        int(bool(re.search(r"\b(which|what value|what k|parameter|exactly|reported)\b",q))),
        int(bool(re.search(r"\b(pde|pinn|gnn|cuda|chebconv|ntk|gradient|equation)\b",q))),
        int(bool(re.search(r"\b(why|how|explain|discuss)\b",q))),
    ]
