"""Rescore an EXISTING retrieval ablation from stored rankings; no API calls.

python -m rag_lab.evaluation.rescore_retrieval \
    --input eval/retrieval_ablation_v2.jsonl \
    --output eval/retrieval_ablation_v2_rescored.jsonl

Never changes the retrieved ranking, context, answer, or stored passage ID.
A source ID prefixed with "unknown:" cannot be regenerated offline because
the ablation JSONL does not contain source passage text. It is flagged.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.knowledge.paper_registry import PaperRegistry

RESCORE_VERSION = "retrieval-v2-rescored-1"
METRICS = ("chunk_precision_at_k", "paper_recall_at_k", "mrr_at_k",
           "ndcg_at_k", "provenance_coverage")


def rescore(input_path, output_path, eval_path):
    source_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("Input and output must be different files")
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; choose another output to preserve experiment history"
        )
    dataset = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    registry = PaperRegistry.from_eval(dataset)
    questions = {q["id"]: q for q in dataset["questions"]}

    rows = []
    missing_labels_before = 0
    fixed_labels = 0
    unresolved_labels = 0
    nonreconstructible_ids = 0
    by_mode = defaultdict(list)
    seen_runs = set()

    with source_path.open(encoding="utf-8") as inp:
        for line_number, line in enumerate(inp, start=1):
            if not line.strip():
                continue
            original = json.loads(line)  # Strict: never silently skip invalid JSONL.
            if original.get("version") != "retrieval-v2":
                raise ValueError(f"Unsupported retrieval version at line {line_number}")
            question_id = original["id"]
            if question_id not in questions:
                raise ValueError(f"Unknown evaluation question {question_id!r}")
            row = dict(original)
            key = (question_id, row.get("mode"), row.get("backend"), row.get("k"),
                   row.get("fetch_k"), row.get("lambda_mult"))
            if key in seen_runs:
                raise ValueError(f"Duplicate ablation run at line {line_number}: {key}")
            seen_runs.add(key)

            fixed_sources = []
            for item in row.get("sources") or []:
                if not isinstance(item, dict):
                    raise ValueError(f"Source is not a dict at line {line_number}")
                entry = dict(item)
                claimed = entry.get("paper_id")
                inferred = registry.resolve(entry.get("source"))
                if claimed and inferred and claimed != inferred:
                    raise ValueError(
                        f"Conflicting paper ID at line {line_number}: "
                        f"existing={claimed}, filename={inferred}"
                    )
                if not claimed:
                    missing_labels_before += 1
                    if inferred:
                        entry["paper_id"] = inferred
                        entry["paper_id_resolution"] = "audited_filename_registry"
                        fixed_labels += 1
                    else:
                        unresolved_labels += 1
                if inferred and str(entry.get("id") or "").startswith("unknown:"):
                    # Passage text is not stored in this retrieval-only JSONL,
                    # so its content-addressed ID cannot be recomputed honestly.
                    entry["passage_id_needs_regeneration"] = True
                    nonreconstructible_ids += 1
                fixed_sources.append(entry)

            row["sources"] = fixed_sources
            score = score_retrieval(
                [item.get("paper_id") for item in fixed_sources],
                questions[question_id].get("source_documents"), k=int(row["k"])
            )
            row.update(score)
            row["evaluation_version"] = RESCORE_VERSION
            rows.append(row)
            by_mode[row.get("mode", "unknown")].append(row)

    # Do not touch the original input. Exclusive creation prevents overwriting
    # a previous corrected run by accident.
    with output_path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_runs": len(rows),
        "missing_paper_ids_before": missing_labels_before,
        "paper_ids_recovered": fixed_labels,
        "paper_ids_still_unknown": unresolved_labels,
        "unchangeable_unknown_passage_ids": nonreconstructible_ids,
        "modes": {},
    }
    for mode, records in sorted(by_mode.items()):
        summary["modes"][mode] = {
            "runs": len(records),
            "scorable": sum(r["chunk_precision_at_k"] is not None for r in records),
            **{metric: (
                    round(mean(r[metric] for r in records
                               if r[metric] is not None), 6)
                    if any(r[metric] is not None for r in records) else None
                ) for metric in METRICS},
        }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate retrieval metrics from saved JSONL (no embeddings or GPT)"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-set", default=Path("eval/eval_set.json"), type=Path)
    args = parser.parse_args()
    report = rescore(args.input, args.output, args.eval_set)
    print(json.dumps(report, indent=2))
    print("Saved corrected runs to:", args.output)


if __name__ == "__main__":
    main()
