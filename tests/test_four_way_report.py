import json
from rag_lab.evaluation.four_way_report import report


def test_four_way_counts_abstention_not_success(tmp_path):
    dataset=tmp_path/"eval.json"
    dataset.write_text(json.dumps({
        "metadata":{"papers":[{"short_id":"xpinn","file":"xpinn.pdf"}]},
        "questions":[{"id":"q","source_documents":["xpinn"]}]
    }))
    gen=tmp_path/"gen.jsonl"
    rows=[
        {"id":"q","backend":backend,"strategy":strategy,
         "contexts":["evidence"] if strategy=="naive" else [],
         "answer":"A" if strategy=="naive" else
                  "Insufficient retrieved evidence to answer this question.",
         "sources":[{"paper_id":"xpinn","source":"xpinn.pdf"}] if strategy=="naive" else [],
         "trace":{"gpt_requests":1 if strategy=="naive" else 0}}
        for backend in ("dense","colbert") for strategy in ("naive","crag")
    ]
    gen.write_text("".join(json.dumps(x)+"\n" for x in rows))
    judge=tmp_path/"judge.jsonl"
    judge.write_text("")
    out=tmp_path/"summary.json"
    result=report(gen,judge,dataset,out)
    assert result["all_four_arms_count"]==1
    assert result["arms"]["dense:crag"]["abstained_or_missing"]==1
    assert result["arms"]["colbert:naive"]["paper_recall_at_4"]==1.0
