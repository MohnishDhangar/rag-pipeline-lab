import pytest
from rag_lab.routing.controller_v2 import decide


def test_four_policy_paths_without_gold_sources():
    assert decide("Explain generic neural networks.").backend=="dense"
    detail=decide("What Laplacian derivative formula is used?",colbert_available=True)
    assert (detail.backend,detail.strategy)==("colbert","naive")
    multi=decide("Both XPINN and SK-PINN compare losses.",colbert_available=True)
    assert (multi.backend,multi.strategy)==("colbert","crag")
    risky=decide("Why does GNS use random-walk noise?",colbert_available=True)
    assert (risky.backend,risky.strategy)==("colbert","crag")


def test_budget_zero_produces_no_grader_requests():
    d=decide("Both XPINN and SK-PINN compare losses.",
             colbert_available=False,max_grader_calls=0)
    assert (d.backend,d.strategy,d.expected_max_grader_calls)==("dense","naive",0)
    with pytest.raises(ValueError):
        decide("Question",max_grader_calls=4)
