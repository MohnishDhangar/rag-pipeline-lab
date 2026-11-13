# CRAG v6.0 — evidence-preserving foundation

This experimental arm is **not** the completed facet-aware CRAG design.
It is a deliberately conservative first step that fixes the destructive
filtering failure mode: it never removes initially retrieved evidence solely
because a passage-level GPT grader calls it ambiguous.

- Retains the original ranked passages unless a named missing paper must
  occupy a slot in the bounded context.
- Adds source-targeted retrieval only for paper identities resolved from the
  question text (never evaluation gold labels).
- Traces correction attempts, initial/final passage counts and unresolved
  named sources.
- Keeps v5 unchanged and provides a separately versioned evaluation arm.
- If a named paper remains missing, generation must explicitly indicate
  unsupported subclaims; this is **not** yet guaranteed by a verifier.
- Does not yet implement facet decomposition, contradiction handling,
  collective evidence grading, or claim-level verification.

Example (requires local corpus, index, Gemini and authorized GPT):

\`\`\`fish
set PY "$HOME/my-jupyter-env/bin/python"
"$PY" -m rag_lab.evaluation.generate \
  --strategies naive crag crag_v6 --backends dense \
  --eval-set eval/heldout_robustness_candidate_30.json \
  --limit 3 --output eval/generated_crag_v6_smoke.jsonl
\`\`\`

**Do not treat these three questions as independent test evidence if used to
tune the controller.** The benchmark JSON must exist locally; it is not
committed to the repository. Evaluate v6 against v5 and naive using identical
question IDs, corpus/index versions, generation model and judge settings.
Run the separate unsupported-claim challenges before promoting v6.

The next milestone is a true facet-level coverage controller with bounded
correction and answer verification, backed by independent references.
