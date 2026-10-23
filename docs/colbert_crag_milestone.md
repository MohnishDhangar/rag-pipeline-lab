# ColBERT + CRAG integration: local execution guide

All four requested priorities now have code paths and mocked tests:
- ColBERT/PyLate: safe builder, mapping verification, live smoke check.
- CRAG: source-aware closed-corpus correction, GPT batch relevance grader,
  source-preserving citations, bounded grading, explicit abstention.
- Evaluation: backend retrieval comparisons and four-arm answer experiment.
- Adaptive v2: text-only cost-capped dense/ColBERT/CRAG controller.

**No real index or OAuth model inference is exercised by GitHub CI.**
Claims about retrieval improvement or scientific accuracy require running the
commands on the local corpus and inspecting their results.

## 1. Verify ColBERT before any GPT calls

Use the SAME interpreter for all commands. For a CPU laptop with 8 GB RAM,
index with a small batch size. First install dependencies:

~~~bash
python -m pip install -e ".[rag,colbert,dev]"
python -m pytest -q

python -m rag_lab.retrieval.colbert_health --root . \
  --expected-chunks 576
python -m rag_lab.retrieval.colbert_health --root . \
  --expected-chunks 576 --query "How does XPINN enforce interface constraints?"
~~~

Without --query, this only validates mapping files and source metadata.
With --query, a REAL PyLate retrieval must return IDs present in both JSON
maps. If there is no index, build it only after ensuring the canonical Chroma
index has 576 passages:

~~~bash
python -m rag_lab.retrieval.export_colbert_leaves --root .
python -m rag_lab.retrieval.colbert_index_build --root . --batch-size 4
python -m rag_lab.retrieval.colbert_health --root . \
  --expected-chunks 576 --query "What is a PDE residual?"
~~~

The builder REFUSES to overwrite an existing index unless explicitly asked.
Back up any existing index before using --overwrite. If your corpus count is
different, change --expected-chunks to its verified count.

Compare retrievers with no GPT requests (dense uses Gemini query embeddings;
ColBERT uses local model inference):

~~~bash
python -m rag_lab.evaluation.retrieval_matrix \
  --backends dense colbert --k 4 --limit 10 \
  --output eval/retrieval_backends_pilot.jsonl
~~~

After a successful pilot, use --limit 63 and a NEW filename for all questions.

## 2. Four-arm end-to-end experiment

The generation CLI now accepts --strategies naive crag on both backends.
The CRAG variants issue up to two **batch** GPT grading requests and one
generation request per answered question; there are no open web fallbacks.
Question/batch references whose identities are unresolved cause abstention.

~~~bash
python -m rag_lab.evaluation.generate \
  --strategies naive crag --backends dense colbert \
  --ids b1_gns_03 b2_cross_004 b3_cross_007 \
  --crag-max-grades 2 --output eval/generated_four_way_pilot.jsonl

python -m rag_lab.evaluation.benchmark judge \
  --input eval/generated_four_way_pilot.jsonl \
  --output eval/judged_four_way_pilot.jsonl

python -m rag_lab.evaluation.four_way_report \
  --generated eval/generated_four_way_pilot.jsonl \
  --judged eval/judged_four_way_pilot.jsonl \
  --output eval/four_way_pilot_summary.json
~~~

For three questions, generation can require up to 12 final-answer GPT calls
plus 12 additional CRAG grading calls (two methods × three questions × two
grades); the judge can require up to 12 more calls. CRAG may abstain and
save these calls. Do not treat only successfully answered questions as the
whole sample: the report includes abstentions.

The evaluation grades are GPT-produced, not independently verified claims.

## 3. Adaptive-v2 four-policy experiment

Controller v2 routes using only the question text, never gold paper IDs,
ground-truth question_type or reference answers.

With ColBERT index present, it can select dense+naive, ColBERT+naive, or
ColBERT+CRAG; without it, risky/cross-paper questions use dense+CRAG.
Set --controller-grader-budget to 0, 1, or 2.

~~~bash
python -m rag_lab.evaluation.generate \
  --strategies adaptive_v2 --backends dense \
  --ids b1_gns_03 b2_cross_004 b3_cross_007 \
  --controller-grader-budget 1 --controller-require-colbert \
  --output eval/generated_adaptive_v2_pilot.jsonl
~~~

Remove --controller-require-colbert to test an explicitly reported dense-only
fallback, but do not compare that result as if ColBERT was used.

## Experimental limits

- The real private ColBERT index and metadata must pass the local live query.
- Grader outputs must be checked for hallucinated or missing evidence and
  calibrated using independently human-reviewed claims.
- Requested-paper aliases are intentionally conservative. An unspecified
  "this paper" or "these three papers in this batch" needs explicit context.
- Historical 63-question results were examined during policy design and
  do not qualify as a held-out validation set.
- No generated-answer quality advantage is claimed without the local
  four-arm benchmark and a separate held-out assessment.
