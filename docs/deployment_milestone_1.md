# Deployment milestone 1: serving/evaluation provenance parity

The dashboard now wraps the selected raw retrieval backend with the **same**
canonical \`ProvenanceStore\` and \`PaperRegistry.from_eval\` used by
\`rag_lab.evaluation.generate\`, before graph augmentation and generation.
The registry uses paper metadata only, never gold question labels.

Why: previously the live dashboard used a raw backend, while the evaluation
pipeline normalized PDF page numbers, source identity and passage IDs.
Different source metadata between demonstration and benchmark is unacceptable.

Security: the dashboard no longer displays raw exception messages in the UI.
The detailed exception remains in local server logs. This is not a complete
security review or multi-user authorization system.

Validation: \`python -m pytest -q tests/test_serving_provenance.py\`.
Full unit CI must pass before merging.

Remaining: extract the rest of the dashboard orchestration into a shared
service, implement index manifests, request limits, real integration tests,
and an appropriate supported production authentication strategy.
