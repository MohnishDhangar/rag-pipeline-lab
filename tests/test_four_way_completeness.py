import json
import pytest
from rag_lab.evaluation.four_way_report import report


def test_report_require_all_four_arms(tmp_path):
    dataset=tmp_path/"eval.json"
    dataset.write_text(json.dumps({
        "metadata":{"papers":[{"short_id":"gns","file":"gns.pdf"}]},
        "questions":[{"id":"q","source_documents":["gns"]}]
    }))
    target=tmp_path/"gen.jsonl"
    target.write_text(json.dumps({
        "id":"q","backend":"dense","strategy":"naive",
        "contexts":["evidence"],"answer":"yes",
        "sources":[{"paper_id":"gns","source":"gns.pdf"}],
        "trace":{"gpt_requests":1}})+"\n")
    judged=tmp_path/"judge.jsonl"
    judged.write_text("")
    out=tmp_path/"result.json"
    with pytest.raises(ValueError,match="Incomplete four-arm"):
        report(target,judged,dataset,out,require_complete=True)
    assert not out.exists()
    with pytest.raises(ValueError,match="Expected/observed question IDs differ"):
        report(target,judged,dataset,out,expected_ids=[],require_complete=False)
