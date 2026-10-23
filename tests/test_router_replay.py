import json
import pytest
from rag_lab.evaluation.router_replay import replay


def fixture_dataset(tmp_path):
    data={
        "metadata":{"papers":[
            {"short_id":"ntk_pinn","title":"When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective","file":None},
            {"short_id":"xpinn","title":"Extended Physics-Informed Neural Networks (XPINNs)","file":None}
        ]},
        "questions":[
            {"id":"one","question_type":"single_document","question":"What is the NTK?","source_documents":["ntk_pinn"]},
            {"id":"two","question_type":"cross_document","question":"Both XPINN and NTK-PINN tackle training cost. Contrast them.","source_documents":["ntk_pinn","xpinn"]},
        ]
    }
    path=tmp_path/"eval.json"
    path.write_text(json.dumps(data))
    return path


def test_offline_replay_uses_only_question_text_for_routing(tmp_path):
    evaluation=fixture_dataset(tmp_path)
    fn=tmp_path/"runs.jsonl"
    output=tmp_path/"replay.json"
    lines=[]
    for id_ in ("one","two"):
        for mode in ("similarity","mmr"):
            source_id="xpinn" if id_=="two" and mode=="mmr" else "ntk_pinn"
            lines.append({"version":"retrieval-v2","id":id_,"mode":mode,"k":4,
                          "sources":[{"paper_id":source_id}],"latency_ms":2.0})
    fn.write_text("".join(json.dumps(line)+"\n" for line in lines))
    res=replay(fn,output,evaluation)
    assert res["overall"]["adaptive"]["n"]==2
    assert res["confusion"]=={"cross_document -> cross_document":1,
                             "single_document -> single_document":1}
    assert next(row for row in res["paired_questions"] if row["id"]=="two")["chosen_mode"]=="mmr"
    assert json.loads(output.read_text())["report_version"]=="router-replay-v1"
    with pytest.raises(FileExistsError):
        replay(fn,output,evaluation)


def test_replay_rejects_missing_pair(tmp_path):
    evaluation=fixture_dataset(tmp_path)
    inp=tmp_path/"in.jsonl"
    inp.write_text(json.dumps({"id":"one","mode":"similarity","k":4,"sources":[],"latency_ms":1})+"\n")
    with pytest.raises(ValueError,match="Expected all questions|Missing paired"):
        replay(inp,tmp_path/"out.json",evaluation)
