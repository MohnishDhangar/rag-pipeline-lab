import json
import pytest

from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.evaluation.rescore_retrieval import rescore


def test_xpinn_alias_matches_multiline_pdf_filename():
    reg = PaperRegistry([{
        "short_id": "xpinn",
        "file": "CiCP_v28i5_2002-2041.pdf",
        "title": "Extended Physics-Informed Neural Networks (XPINNs)"
    }])
    stored = ("/home/user/papers/XPINNs - A Generalized Space-Time Domain Decomposition\n"
              "Based Deep Learning Framework for Nonlinear\n"
              "Partial Differential Equations.pdf")
    assert reg.resolve(stored) == "xpinn"


def test_rescore_fixes_labels_and_metrics_without_touching_existing_experiment(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({
        "metadata": {"papers": [{"short_id": "xpinn", "file": None,
                                 "title": "Extended Physics-Informed Neural Networks (XPINNs)"}]},
        "questions": [{"id": "q", "source_documents": ["xpinn"]}]
    }))
    filename = "XPINNs - A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations.pdf"
    input_path = tmp_path / "old.jsonl"
    output_path = tmp_path / "new.jsonl"
    old = {"id": "q", "version": "retrieval-v2", "mode": "similarity",
           "backend": "dense", "k": 4, "fetch_k": None, "lambda_mult": None,
           "sources": [{"id": "unknown:abcd", "paper_id": None,
                        "source": "/paper-dir/" + filename}],
           "chunk_precision_at_k": None, "mrr_at_k": None}
    input_path.write_text(json.dumps(old) + "\n")
    report = rescore(input_path, output_path, eval_path)
    row = json.loads(output_path.read_text())
    assert json.loads(input_path.read_text()) == old
    assert report["paper_ids_recovered"] == 1
    assert report["paper_ids_still_unknown"] == 0
    assert row["sources"][0]["paper_id"] == "xpinn"
    assert row["sources"][0]["id"] == "unknown:abcd"
    assert row["sources"][0]["passage_id_needs_regeneration"] is True
    assert row["provenance_coverage"] == 1.0
    assert row["chunk_precision_at_k"] == .25
    assert row["paper_recall_at_k"] == 1.0
    assert row["mrr_at_k"] == 1.0
    assert row["evaluation_version"] == "retrieval-v2-rescored-1"
    with pytest.raises(FileExistsError):
        rescore(input_path, output_path, eval_path)


def test_rescore_refuses_conflicting_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({
        "metadata": {"papers": [{"short_id": "xpinn", "file": None,
                                 "title": "Extended Physics-Informed Neural Networks (XPINNs)"}]},
        "questions": [{"id": "q", "source_documents": ["xpinn"]}]
    }))
    input_path = tmp_path / "bad.jsonl"
    output_path = tmp_path / "no-output.jsonl"
    input_path.write_text(json.dumps({
        "id": "q", "version": "retrieval-v2", "mode": "mmr", "k": 4,
        "sources": [{"paper_id": "ntk_pinn", "source":
            "XPINNs - A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations.pdf"}]
    })+"\n")
    with pytest.raises(ValueError, match="Conflicting paper ID"):
        rescore(input_path, output_path, eval_path)
    assert not output_path.exists()
