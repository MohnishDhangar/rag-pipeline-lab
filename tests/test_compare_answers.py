import json
import pytest

from rag_lab.evaluation.benchmark import row_fingerprint
from rag_lab.evaluation.compare_answers import compare
from rag_lab.evaluation.judge import RUBRIC_VERSION


def test_judge_comparison_joins_identical_generation_fingerprints(tmp_path):
    generated=tmp_path/"generated.jsonl"
    judged=tmp_path/"judged.jsonl"
    output=tmp_path/"comparison.json"
    rows=[{"id":"q","strategy":mode,"backend":"dense",
           "question":"Q","answer":"An answer","contexts":["evidence"],
           "sources":[{"id":"paper.pdf"}]}
          for mode in ("naive","adaptive")]
    generated.write_text("".join(json.dumps(r)+"\n" for r in rows))
    scores=[]
    for i,row in enumerate(rows):
        scores.append({"id":"q","strategy":row["strategy"],"backend":"dense",
                       "judge_version":RUBRIC_VERSION,
                       "input_fingerprint":row_fingerprint(row),
                       "faithfulness":0.6+i*0.2,"answer_correctness":.7,
                       "answer_relevance":.8,"citation_support":.9})
    judged.write_text("".join(json.dumps(r)+"\n" for r in scores))
    result=compare(generated,judged,output)
    assert result["paired_n"] == 1
    assert result["means"]["faithfulness"]["delta"] == .2
    assert json.loads(output.read_text())["paired_n"] == 1
    with pytest.raises(FileExistsError):
        compare(generated,judged,output)


def test_stale_fingerprint_cannot_be_used(tmp_path):
    generated=tmp_path/"generated.jsonl"
    judged=tmp_path/"judged.jsonl"
    output=tmp_path/"comparison.json"
    rows=[{"id":"q","strategy":"naive","backend":"dense","question":"Q",
           "answer":"A","contexts":["evidence"]},
          {"id":"q","strategy":"adaptive","backend":"dense","question":"Q",
           "answer":"B","contexts":["evidence"]}]
    generated.write_text("".join(json.dumps(r)+"\n" for r in rows))
    scores=[{"id":"q","strategy":r["strategy"],"backend":"dense",
             "judge_version":RUBRIC_VERSION,"input_fingerprint":"stale",
             "faithfulness":1,"answer_correctness":1,"answer_relevance":1,
             "citation_support":1} for r in rows]
    judged.write_text("".join(json.dumps(s)+"\n" for s in scores))
    result=compare(generated,judged,output)
    assert result["paired_n"] == 0
    assert len(result["unmatched_judge_rows"]) == 2
