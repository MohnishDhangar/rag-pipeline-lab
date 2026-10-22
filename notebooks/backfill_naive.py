"""
One-time reconciliation: copies the "naive" and "colbert" rows already
paid for in eval_results.jsonl into eval_results_backend.jsonl's
(strategy, backend, id) schema, so 09_colbert_generation.ipynb's
checkpoint sees them as already done and never re-spends quota
recomputing them.

NaiveStrategy -- it's naive-style generation on the colbert backend, not
a different strategy. So:
    eval_results.jsonl  strategy="naive"    -> backend="dense"
    eval_results.jsonl  strategy="colbert"  -> backend="colbert"
both map onto strategy="naive" in the backend-crossed schema.

Run this ONCE, before (re)running 09_colbert_generation.ipynb's main
loop. Safe to run more than once -- it skips any (strategy, backend, id)
already present in eval_results_backend.jsonl rather than duplicating.
"""
import json
from pathlib import Path

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
print(f"project_root resolved to: {project_root}")  # sanity-check this before trusting the paths below

SOURCE = project_root / "eval" / "eval_results.jsonl"
DEST = project_root / "eval" / "eval_results_backend.jsonl"

STRATEGY_TO_BACKEND = {
    "naive": "dense",
    "colbert": "colbert",
}

# What's already in the destination, so this is idempotent.
existing = set()
if DEST.exists():
    with open(DEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            existing.add((row.get("strategy"), row.get("backend"), row.get("id")))

backfilled = 0
skipped_already_present = 0
skipped_no_answer = 0

with open(SOURCE) as f:
    source_rows = [json.loads(line) for line in f if line.strip()]

with open(DEST, "a") as out:
    for row in source_rows:
        src_strategy = row.get("strategy")
        if src_strategy not in STRATEGY_TO_BACKEND:
            continue  # multi_query/rag_fusion/hyde/step_back/decomposition stay untouched -- not redundant
        if row.get("answer") is None:
            skipped_no_answer += 1
            continue

        backend = STRATEGY_TO_BACKEND[src_strategy]
        key = ("naive", backend, row["id"])
        if key in existing:
            skipped_already_present += 1
            continue

        out.write(json.dumps({
            "strategy": "naive",
            "backend": backend,
            "id": row["id"],
            "question": row["question"],
            "question_type": row.get("question_type"),
            "ground_truth": row.get("ground_truth"),
            "answer": row["answer"],
            "contexts": row.get("contexts"),
            "error": None,
        }) + "\n")
        existing.add(key)
        backfilled += 1

print(f"Backfilled {backfilled} rows into {DEST.name}")
print(f"Skipped {skipped_already_present} already present, {skipped_no_answer} with no answer")
