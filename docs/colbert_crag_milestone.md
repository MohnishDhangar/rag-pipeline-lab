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


## Repair an existing ColBERT index missing id_to_metadata.json

If `colbert_health` fails on missing `data/id_to_metadata.json`, do not
immediately rebuild PLAID. The exporter previously constructed
`GoogleGenerativeAIEmbeddings` just to read local Chroma passages, which
unnecessarily required `GOOGLE_API_KEY`. The repaired exporter reads local
Chroma via `chromadb.PersistentClient` and makes **no API requests**:

~~~fish
cd ~/Codes/rag-lab
set PY "$HOME/my-jupyter-env/bin/python"

# This preserves any existing data/colbert_index and data/id_to_text.json.
"$PY" -m rag_lab.retrieval.export_colbert_leaves --root .

# Check the ID/text/metadata counts and source aliases.
"$PY" -m rag_lab.retrieval.colbert_health --root . --expected-chunks 576

# Exercise the actual PyLate retrieval model, not just the mapping.
"$PY" -m rag_lab.retrieval.colbert_health --root . --expected-chunks 576 \
  --query "How does XPINN enforce interface constraints?"
~~~

The exporter aligns old ColBERT chunk IDs to Chroma via EXACT stored text,
not an assumed positional order. If duplicate text appears in multiple PDFs
or pages with conflicting metadata, it stops: text-only mappings cannot
determine which PDF/page the historical PLAID index originally used.
Likewise, a corpus mismatch or stale `leaf_texts.json` stops rather than
overwriting existing index files. Restore older mappings or deliberately
build a new, separately backed-up index in that situation. The metadata
export does not generate a manifest for a pre-existing index, because it
cannot prove which embeddings PLAID stored. A missing manifest thus reports
`unverified_no_manifest`, not `matched`; only the current builder writes a
manifest after successfully indexing.


## Phase 4: controlled three-question CRAG answer-quality pilot (2026-10)

Historical 63-question Dense/ColBERT retrieval results were inspected. The next
experiment explicitly targets three distinct diagnostic cases:

- \`b1_gns_03\`: Dense retrieved the wrong scientific paper; ColBERT recovered GNS.
- \`b1_cross_01\`: two explicitly identified papers with different training analyses.
- \`b2_cross_003\`: two explicitly identified papers with technical derivative
  implementations; ColBERT recovered both papers where Dense missed one.

Do not substitute \`b2_cross_004\` as a positive comparison case without
reformulating its ambiguous "this batch" reference. CRAG should abstain on
unresolved external context. Keep original benchmark questions unchanged for
this experiment, and do not leak gold \`source_documents\` to runtime retrieval.

### Local Fish shell

~~~fish
cd ~/Codes/rag-lab
git pull --ff-only origin feature/adaptive-rag-gpt-oauth
set PY "$HOME/my-jupyter-env/bin/python"

# PyLate already works in this interpreter. Install the RAG/GPT dependencies,
# not the optional ColBERT extra or a new PyLate index.
"$PY" -m pip install -e ".[rag,dev]"
"$PY" -m pip check

# Set GOOGLE_API_KEY/GEMINI_API_KEY and RAG_GPT_MODEL in this Fish session.
# Never paste tokens into logs or a GitHub issue.
"$PY" -m rag_lab.llms.cli models
"$PY" -m rag_lab.evaluation.crag_preflight --root .
~~~

The preflight checks local populated Chroma, existing ColBERT mappings, PyJWT,
PyLate, question IDs, environment-variable presence, and the presence of GPT
model selection. It neither loads PLAID weights nor requests an access token.

~~~fish
# Run the four arms over the identical three questions (12 records).
"$PY" -m rag_lab.evaluation.generate \
  --strategies naive crag --backends dense colbert \
  --ids b1_gns_03 b1_cross_01 b2_cross_003 \
  --crag-max-grades 2 --crag-pool-k 28 \
  --output eval/generated_crag_v3_pilot.jsonl

# Judge ONLY answered rows, keeping abstentions in the generation data.
"$PY" -m rag_lab.evaluation.benchmark judge \
  --input eval/generated_crag_v3_pilot.jsonl \
  --output eval/judged_crag_v3_pilot.jsonl

# Fail if any expected question or four-arm record is missing.
"$PY" -m rag_lab.evaluation.four_way_report \
  --generated eval/generated_crag_v3_pilot.jsonl \
  --judged eval/judged_crag_v3_pilot.jsonl \
  --expected-ids b1_gns_03 b1_cross_01 b2_cross_003 \
  --require-complete \
  --output eval/four_way_crag_v3_summary.json
~~~

A complete run has exactly twelve distinct \`(backend, strategy, question ID)\`
records even if some answers are abstentions. This requires up to 12 final
answer GPT requests and 12 additional batch grading requests (CRAG only);
the judging stage can require at most 12 GPT requests. If the GPT provider
fails, checkpoints preserve completed runs; rerun with identical configuration.
The output filename is new because the corrective implementation has changed.

To inspect the status/abstention distribution without any model requests:

~~~fish
"$PY" -c 'import json,collections; p="eval/generated_crag_v3_pilot.jsonl"; a=[json.loads(x) for x in open(p) if x.strip()]; print("rows",len(a)); print(collections.Counter((r["backend"],r["strategy"],r.get("generation_status")) for r in a))'
~~~

**Important limitations.** The historical questions influenced development
and are not held-out evidence. GPT judging should be corroborated by human
passage review. Source diversity is insufficient to establish passage-level
support. The 576-chunk legacy ColBERT index was successfully checked by a
live retrieval query but retains \`unverified_no_manifest\` until reindexed:
do not create an unearned fingerprint manifest for the old PLAID index.
