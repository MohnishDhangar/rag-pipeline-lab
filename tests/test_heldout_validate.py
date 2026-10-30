import json
import pytest
from rag_lab.evaluation.heldout_validate import make_template, check


def development(tmp_path):
    data={"metadata":{"papers":[{"short_id":"gns","file":"gns.pdf"},
                                {"short_id":"xpinn","file":"xpinn.pdf"}]},
          "questions":[{"id":"old_q1","question":"Why do authors inject velocity noise during training?",
                        "source_documents":["gns"]}]}
    path=tmp_path/"dev.json"
    path.write_text(json.dumps(data))
    return path


def test_holdout_template_is_empty_and_carries_registry(tmp_path):
    template=make_template(development(tmp_path))
    assert len(template["metadata"]["papers"])==2
    assert template["questions"]==[]
    assert not template["metadata"]["human_reference_reviewed"]


def test_heldout_check_prohibits_development_questions_and_vague_scope(tmp_path):
    dev=development(tmp_path)
    template=make_template(dev)
    candidate=tmp_path/"holdout.json"
    row={
        "id":"h1","question":"What is the mathematical purpose of interface continuity conditions in XPINN decomposition?",
        "source_documents":["xpinn"],"question_type":"single_document",
        "ground_truth":"Interface conditions link neighboring XPINN subdomain solutions and enforce a consistent approximation across boundaries."
    }
    template["questions"]=[row]
    candidate.write_text(json.dumps(template))
    report=check(dev,candidate,minimum=1)
    assert report["count"]==1
    assert report["status"]=="structurally_disjoint_unreviewed"
    template["questions"][0]["id"]="old_q1"
    candidate.write_text(json.dumps(template))
    with pytest.raises(ValueError,match="development-set ID"):
        check(dev,candidate,minimum=1)
    template["questions"][0]["id"]="h2"
    template["questions"][0]["question"]="Compare all three papers in this batch regarding interface constraints."
    candidate.write_text(json.dumps(template))
    with pytest.raises(ValueError,match="unspecified paper/batch"):
        check(dev,candidate,minimum=1)


def test_holdout_rejects_question_text_near_duplicate(tmp_path):
    dev=development(tmp_path)
    candidate=tmp_path/"holdout.json"
    template=make_template(dev)
    template["questions"]=[{
        "id":"h2","question":"Why do authors inject velocity noise during training?",
        "source_documents":["gns"],"question_type":"single_document",
        "ground_truth":"The paper studies robustness to accumulated autoregressive model error during rollouts."
    }]
    candidate.write_text(json.dumps(template))
    with pytest.raises(ValueError,match="too similar"):
        check(dev,candidate,minimum=1)
