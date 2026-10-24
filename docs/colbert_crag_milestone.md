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
  --require-complete --require-judged \
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


## CRAG v4: source-filtered ColBERT correction after analyzing v3

The 3-question v3 trial returned just one scientifically grounded CRAG answer per
backend (the GNS random-walk noise question); both multi-paper questions
abstained. Although GPT judge correctness was 1.0 conditional on answering,
**answer coverage was only 1/3**, so conditional score means are misleading.
The v3 dense naive GNS answer cited the **wrong** MeshGraphNets paper despite a
high citation-support rating. Grounded source identity must be checked.

v4 addresses a concrete implementation defect: the ColBERT retriever now maps
\`source\` metadata to PyLate's native \`subset=[document IDs]\` search instead
of globally retrieving top-28 and filtering them afterward. Source-aware
search can target explicitly named papers in ColBERT just as it can in Chroma.
It also logs per-passage grader decisions to explain which paper had merely
retrieved content versus verified answer-bearing quotes.

**Avoid reusing \`eval/generated_crag_v3_pilot.jsonl\` as v4 output.**
The two CRAG algorithms differ. Preserve v3 files untouched.

### Validate actual PyLate source restriction before spending GPT calls

~~~fish
cd ~/Codes/rag-lab
git pull --ff-only origin feature/adaptive-rag-gpt-oauth
set PY "$HOME/my-jupyter-env/bin/python"

"$PY" -m rag_lab.retrieval.colbert_health \
  --root . --expected-chunks 576 \
  --query "How does the original Raissi PINN paper approach training difficulties?" \
  --paper-id orig_pinn --k 4

"$PY" -m rag_lab.retrieval.colbert_health \
  --root . --expected-chunks 576 \
  --query "How is the Laplacian approximated on an irregular triangular mesh?" \
  --paper-id pignn_al --k 4
~~~

Both commands should report the requested paper ID for EVERY returned passage.
These are CPU-local retrieval queries; no Gemini or GPT calls.

### Run ONLY the six new CRAG v4 arms

Keep the same \`RAG_GPT_MODEL=gpt-6-sol\` model (or the exact model used in v3)
for a matched, reusable baseline.

~~~fish
"$PY" -m rag_lab.evaluation.crag_preflight --root .

"$PY" -m rag_lab.evaluation.generate \
  --strategies crag --backends dense colbert \
  --ids b1_gns_03 b1_cross_01 b2_cross_003 \
  --crag-max-grades 2 --crag-pool-k 28 \
  --output eval/generated_crag_v4_pilot.jsonl

"$PY" -m rag_lab.evaluation.benchmark judge \
  --input eval/generated_crag_v4_pilot.jsonl \
  --output eval/judged_crag_v4_pilot.jsonl
~~~

Unlike repeating all four arms, this does NOT regenerate the six naive answers.
The conservative worst case is 18 GPT requests for corrective grading and
answering, plus up to 6 GPT judge requests. Abstention reduces that number.

### Merge the exact six naive v3 baselines and judge rows, offline

~~~fish
"$PY" -m rag_lab.evaluation.assemble_crag_pilot \
  --baseline-generated eval/generated_crag_v3_pilot.jsonl \
  --baseline-judged eval/judged_crag_v3_pilot.jsonl \
  --candidate-generated eval/generated_crag_v4_pilot.jsonl \
  --candidate-judged eval/judged_crag_v4_pilot.jsonl \
  --generated eval/generated_crag_v4_four_way.jsonl \
  --judged eval/judged_crag_v4_four_way.jsonl \
  --ids b1_gns_03 b1_cross_01 b2_cross_003

"$PY" -m rag_lab.evaluation.four_way_report \
  --generated eval/generated_crag_v4_four_way.jsonl \
  --judged eval/judged_crag_v4_four_way.jsonl \
  --expected-ids b1_gns_03 b1_cross_01 b2_cross_003 \
  --require-complete --require-judged \
  --output eval/four_way_crag_v4_summary.json
~~~

The merger requires identical model slug, question text and evaluation
annotations across v3 naive / v4 CRAG outputs, checks every judge fingerprint,
rejects prior v3 CRAG rows, and never reruns inference.

CRAG v4 is an **experimental fix**, not a proven improvement. It can still
abstain when the correct passage is absent or the GPT grader rejects it.
The grade trace now distinguishes those failure causes.

The experiment remains on the previously inspected questions; a separate,
unseen and human-verified sample is required before scientific reliability
or performance claims.


## CRAG v5: bounded quote-recovery experiment (after v4 pilot)

The 2026-10 pilot generated 6 CRAG v4 records. Answer coverage rose
from v3's **2/6** to **4/6**, but the GNS ColBERT arm regressed to abstention
despite having an initially retrieved GNS passage. The v4 trace contained
**7 passages** that the GPT relevance grader labelled \`correct\` but whose
quotes **did not exactly occur** in the corresponding source text. The
exact-quote gate correctly rejected them; the v4 corrective step then
discarded these passages, often losing strong evidence.

Other issues remain: the comparative CRAG v4 answers have incomplete
scientific coverage, with judged correctness of just 0.40–0.56 for the
answered cross-paper cases. We must not inflate correctness by averaging
only answered questions. The three questions are already tuning examples,
**not held-out evidence**.

### New v5 behavior

- First-grade \`correct\` claims with **invalid** exact quotes are still
  rejected. Up to **two** of these previously unverified passages may be
  reconsidered in the second (and last) grading call, rather than discarded.
- The *corrective grading* candidate set includes up to **8 new** passages
  (from an initial pool of up to 28), plus up to 2 quote rechecks; the *final
  context* remains capped at 4 exact-quote-verified passages.
- The grader is explicitly instructed to copy a character-for-character
  substring, preserving PDF line breaks. Quote previews are logged for
  postmortem inspection; **unverifiable quotations remain unacceptable**.
- Both GPT grading-call budget (maximum 2) and provenance guarantees remain
  unchanged. Additional passages can increase *input tokens*, so compare
  token use and latency as well as call counts.

### Launch a NEW v5 experiment without overwriting any v3/v4 data

~~~fish
cd ~/Codes/rag-lab
git pull --ff-only origin feature/adaptive-rag-gpt-oauth
set PY "$HOME/my-jupyter-env/bin/python"
"$PY" -m pytest -q
"$PY" -m rag_lab.evaluation.crag_preflight --root .

# Keep exactly the same RAG_GPT_MODEL as in v3 and v4.
"$PY" -m rag_lab.evaluation.generate \
  --strategies crag --backends dense colbert \
  --ids b1_gns_03 b1_cross_01 b2_cross_003 \
  --crag-max-grades 2 --crag-pool-k 28 \
  --output eval/generated_crag_v5_pilot.jsonl

"$PY" -m rag_lab.evaluation.benchmark judge \
  --input eval/generated_crag_v5_pilot.jsonl \
  --output eval/judged_crag_v5_pilot.jsonl

"$PY" -m rag_lab.evaluation.assemble_crag_pilot \
  --baseline-generated eval/generated_crag_v3_pilot.jsonl \
  --baseline-judged eval/judged_crag_v3_pilot.jsonl \
  --candidate-generated eval/generated_crag_v5_pilot.jsonl \
  --candidate-judged eval/judged_crag_v5_pilot.jsonl \
  --generated eval/generated_crag_v5_four_way.jsonl \
  --judged eval/judged_crag_v5_four_way.jsonl \
  --crag-version closed-corpus-crag-v5 \
  --ids b1_gns_03 b1_cross_01 b2_cross_003

"$PY" -m rag_lab.evaluation.four_way_report \
  --generated eval/generated_crag_v5_four_way.jsonl \
  --judged eval/judged_crag_v5_four_way.jsonl \
  --expected-ids b1_gns_03 b1_cross_01 b2_cross_003 \
  --require-complete --require-judged \
  --output eval/four_way_crag_v5_summary.json
~~~

The old naive baseline rows are reused only if the model slug, question
text, and evaluation annotations are unchanged and matching judge
fingerprints are present. Do not merge v4 CRAG rows as if they were v5.

A useful *engineering regression* threshold is that the ColBERT GNS arm
does not lose its initially relevant passage solely because the grader
failed to copy an exact quote. A useful *research* threshold is reliable
completeness and accuracy on **new**, independently labelled and
human-reviewed examples. Passing the engineering threshold does not
establish generalization or prove that CRAG outperforms naive retrieval.


## Phase 5 — independent evaluation protocol, after v5 pilot

CRAG v5 development results on the same three repeatedly studied questions:

| Configuration | Answered | Mean GPT-judged correctness (answered) | Mean latency |
| --- | ---: | ---: | ---: |
| Dense + naive | 3/3 | 0.567 | 9.28 s |
| ColBERT + naive | 3/3 | 0.607 | 6.06 s |
| Dense + CRAG v5 | 2/3 | 0.530 | 30.42 s |
| ColBERT + CRAG v5 | 3/3 | 0.693 | 29.98 s |

**These are development-set scores and should not be used as evidence of
generalization.** The dense CRAG GNS response used a MeshGraphNets passage to
answer a GNS-specific question, scored just 0.30 correctness, yet its
citation-support score was 1.00. A citation can be syntactically and
contextually valid while citing the WRONG research work. Multi-paper answers
still omit key scientific mechanisms, despite full source coverage.

The default for new experiments is therefore unchanged. Retain all four arms
until a new, human-verified question set shows a benefit that justifies
CRAG's higher latency and extra GPT grading calls.

### Build a new, pre-registered held-out evaluation split

Do NOT re-label the existing 63 benchmark questions as held-out. Those
questions and their paper annotations already informed algorithm development.

Create a new file containing **at least 12 independently authored,
human-verified** questions, including a useful spread across single-paper,
cross-paper, numerical, derivation, ablation, and failure-analysis tasks.
The following code creates a blank template retaining the SAME 11-paper
canonical registry as your Chroma/ColBERT indexes:

~~~fish
cd ~/Codes/rag-lab
git pull --ff-only origin feature/adaptive-rag-gpt-oauth
set PY "$HOME/my-jupyter-env/bin/python"

"$PY" -m rag_lab.evaluation.heldout_validate init \
  --development eval/eval_set.json \
  --output eval/heldout_eval_set.json
~~~

Author NEW entries inside \`questions\`, each with:
\`id\` (e.g. \`h1_001\`), \`question\`, \`question_type\`
(\`single_document\` or \`cross_document\`),
\`source_documents\` (canonical paper IDs, used ONLY by offline scoring)
and \`ground_truth\` (independently reviewed against the original PDF).
Mark \`metadata.human_reference_reviewed\` true only when actually reviewed.
Questions must be standalone: no "this batch" or undefined "the paper."

Before any GPT retrieval or judging, run the validator and RECORD its
\`canonical_sha256\`. It checks ID disjointness, exact/near-duplicate
development questions, schema, paper IDs, source cardinality and unspecified
context. It does NOT establish independent authorship or truth accuracy.

~~~fish
"$PY" -m rag_lab.evaluation.heldout_validate check \
  --development eval/eval_set.json \
  --candidate eval/heldout_eval_set.json \
  --minimum 12

sha256sum eval/heldout_eval_set.json
~~~

Do not edit the held-out questions or reference answers after seeing model
outputs; changes invalidate the preregistration.

### Run full matched four-way trial ONCE on the new split

~~~fish
"$PY" -m rag_lab.evaluation.generate \
  --eval-set eval/heldout_eval_set.json \
  --strategies naive crag --backends dense colbert \
  --crag-max-grades 2 --crag-pool-k 28 \
  --output eval/heldout_generated_crag_v5.jsonl

"$PY" -m rag_lab.evaluation.benchmark judge \
  --eval eval/heldout_eval_set.json \
  --input eval/heldout_generated_crag_v5.jsonl \
  --output eval/heldout_judged_crag_v5.jsonl

"$PY" -m rag_lab.evaluation.four_way_report \
  --eval-set eval/heldout_eval_set.json \
  --generated eval/heldout_generated_crag_v5.jsonl \
  --judged eval/heldout_judged_crag_v5.jsonl \
  --require-complete --require-judged \
  --output eval/heldout_four_way_crag_v5_summary.json
~~~

For 12 questions, expect **48 generation records**, plus judge records for
answered rows. The four-arm generation has up to 72 GPT requests:
24 naive answers + 24 CRAG final answers + 24 CRAG batch grades; with two
grades, CRAG alone may need up to 48 grading requests instead, so the true
upper bound is **96 generation calls** (24 naive + 24 CRAG answer + 48
CRAG grading) and **48 judge calls**. Fewer if CRAG abstains. Choose an
appropriate smaller **exploratory, non-held-out** batch first if required by
budget and re-freeze a NEW independent final split for confirmation.

The report's GPT judge scores alone are NOT independent human validation.
Human-review at least the incorrect-source answers, both-paper coverage,
the exactness of evidence, scientific adequacy and the abstention decisions.
Report answer rate, source recall, scientific correctness, abstention-adjusted
quality, p50/p95 latency, and actual per-arm model calls.
