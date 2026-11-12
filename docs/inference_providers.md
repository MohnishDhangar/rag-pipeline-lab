# Inference providers

The package supports ChatGPT OAuth, Groq and Gemini. There is no automatic
provider/model fallback: an interrupted experiment must resume with the same
configuration. Groq/Gemini API requests do not consume ChatGPT subscription
usage. They have independent rate limits and billing.

Set API keys outside source files: GROQ_API_KEY, GEMINI_API_KEY (or
GOOGLE_API_KEY). OAuth continues to use its existing authorization flow.
RAG_LLM_PROVIDER selects a default; --provider overrides it. Model defaults:
Groq openai/gpt-oss-120b; Gemini RAG_GEMINI_MODEL (explicit); OAuth RAG_GPT_MODEL.
RAG_JUDGE_PROVIDER and RAG_JUDGE_MODEL independently configure judging.
RAG_LLM_CACHE is optional, provider-namespaced local SQLite caching for API
providers; RAG_GPT_CACHE remains the OAuth cache. Keys are never stored in caches
or outputs. Cache stores research prompts' responses; keep it private.

Test a connection (one inference request):
```fish
"$PY" -m rag_lab.llms.cli test --provider groq --model openai/gpt-oss-120b
```

Run a new experiment (choose a filename not used by previous providers/models):
```fish
"$PY" -m rag_lab.evaluation.generate --provider groq --model openai/gpt-oss-120b \
    --strategies naive crag crag_v6 --backends dense --limit 3 \
    --eval-set eval/heldout_robustness_candidate_30.json \
    --output eval/generated_groq120b_paired3.jsonl
```
Use the existing correct index and local registry. The remote branch predates
the local BGE/corpus16 refactor; see migration instructions below. Do not run
against a Gemini-embedded index with BGE embeddings or vice versa.

Judge with a separate model and new output:
```fish
"$PY" -m rag_lab.evaluation.benchmark judge --judge-provider gemini \
    --judge-model "$RAG_GEMINI_MODEL" --input eval/generated_groq120b_paired3.jsonl \
    --output eval/judged_groq120b_gemini.jsonl
```

Notebook interface:
```python
from rag_lab.llms.factory import build_chat_model
llm = build_chat_model(provider="groq", model="openai/gpt-oss-120b")
```
Historical notebook experiments retain their original SDK code, model choices
and safe getpass prompts; prefer shared pipeline for new controlled experiments.
Run notebooks from the repository environment. Never commit real keys.

Groq uses non-streaming JSON Schema best-effort output plus Pydantic validation;
Gemini uses generateContent JSON Schema plus Pydantic validation. Neither
schema adherence nor model judging establishes scientific correctness.
No automatic retries or provider switching on 429: preserve completed rows and
retry the same experiment after the provider's limit permits. Max output is
4096 tokens for API providers; incomplete responses fail rather than being
silently accepted. Provider-specific API and model availability must be tested
with a real key. No live requests are made in unit CI.

## Local checkpoint integration

If your checkout already has rag_lab/core/factory.py and
rag_lab/evaluation/experiment.py, do not replace these or merge older generator
code blindly. Fetch this branch, retrieve the new provider files and the
migration utility, and run the utility with --check first. See its --help.
It applies narrowly scoped changes to the local manifest-bound generator,
preserving the BGE index and all existing experiment identity checks.
