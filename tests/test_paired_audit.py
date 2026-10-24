"""Make it impossible to mistake strong GPT correctness for proper provenance."""
import pytest
from rag_lab.evaluation.paired_audit import audit


def report():
    rows={}
    for qid,correct in (("good",True),("wrong",False)):
        rows[qid]={}
        for backend in ("dense","colbert"):
            for strategy in ("naive","crag"):
                key=f"{backend}:{strategy}"
                rows[qid][key]={
                    "id":qid,"answered":True,"paper_recall_at_4":1.0 if correct else 0.0,
                    "judge":{"answer_correctness":.99},
                    "latency_ms":100 if strategy=="naive" else 300,
                }
    return {"per_question":rows,"incomplete_question_ids":[]}


def test_high_correctness_does_not_count_as_source_grounded():
    result=audit(report(),{"good":"single_document","wrong":"cross_document"},
                 iterations=200,seed=1)
    assert result["questions"]==2
    assert result["arms"]["dense:naive"]["correctness_zero_abstentions"]==.99
    assert result["arms"]["dense:naive"]["fully_grounded_answer_rate"]==.5
    assert result["arms"]["dense:naive"]["correctness_and_complete_source_proxy"]==.495
    assert result["arms"]["dense:naive"]["wrong_source_high_correctness_count"]==1
    assert result["wrong_source_high_correctness_cases"]["dense:naive"][0]["id"]=="wrong"
    assert result["subgroups"]["cross_document"]["dense:naive"]["complete_source_coverage"]==0


def test_missing_judge_and_incomplete_arms_fail():
    s=report()
    s["per_question"]["good"]["dense:crag"]["judge"]=None
    with pytest.raises(ValueError,match="unjudged"):
        audit(s,iterations=50)
    s=report()
    del s["per_question"]["wrong"]["colbert:naive"]
    with pytest.raises(ValueError,match="four-way"):
        audit(s,iterations=50)
