import json
import pytest
from rag_lab.evaluation.assemble_crag_pilot import assemble
from rag_lab.evaluation.identity import row_fingerprint, RUBRIC_VERSION


def write(path,items):
    path.write_text("".join(json.dumps(x)+"\n" for x in items))


def row(id,backend,strategy,answered=True):
    return {
        "id":id,"backend":backend,"strategy":strategy,
        "question":"scientific question","model":"test-gpt",
        "source_documents":["gns"],
        "answer":"Scientific answer" if answered else
                 "Insufficient retrieved evidence to answer this question.",
        "contexts":["verbatim evidence"] if answered else [],
        "sources":[{"paper_id":"gns"}] if answered else [],
        "trace":{"crag":{"version":"closed-corpus-crag-v4",
                           "action":"correct" if answered else "abstain"}}
                 if strategy=="crag" else {},
    }


def test_reuse_old_naive_without_new_model_calls(tmp_path):
    bg,bj,cg,cj=[tmp_path/name for name in ("bg","bj","cg","cj")]
    base=[row("q",b,"naive") for b in ("dense","colbert")]
    new=[row("q","dense","crag"),row("q","colbert","crag",answered=False)]
    write(bg,base);write(cg,new)
    write(bj,[{"id":"q","backend":x["backend"],"strategy":"naive",
               "judge_version":RUBRIC_VERSION,"input_fingerprint":row_fingerprint(x),
               "answer_correctness":.6} for x in base])
    write(cj,[{"id":"q","backend":"dense","strategy":"crag",
               "judge_version":RUBRIC_VERSION,
               "input_fingerprint":row_fingerprint(new[0]),"answer_correctness":.9}])
    outg,outj=tmp_path/"merged.jsonl",tmp_path/"merged_j.jsonl"
    info=assemble(bg,bj,cg,cj,outg,outj,ids=["q"])
    assert info["generation_rows"]==4
    assert info["judged_rows"]==3
    assert len(outg.read_text().splitlines())==4
    assert len(outj.read_text().splitlines())==3
    with pytest.raises(FileExistsError):
        assemble(bg,bj,cg,cj,outg,outj,ids=["q"])


def test_do_not_mix_older_crag_algorithm_with_new_report(tmp_path):
    bg,bj,cg,cj=[tmp_path/name for name in ("bg","bj","cg","cj")]
    base=[row("q",b,"naive") for b in ("dense","colbert")]
    wrong=[row("q",b,"crag") for b in ("dense","colbert")]
    wrong[0]["trace"]["crag"]["version"]="closed-corpus-crag-v3"
    write(bg,base);write(cg,wrong)
    bj.write_text("");cj.write_text("")
    with pytest.raises(ValueError,match="Expected v4"):
        assemble(bg,bj,cg,cj,tmp_path/"g",tmp_path/"j",ids=["q"])
