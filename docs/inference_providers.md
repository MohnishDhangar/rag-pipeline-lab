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
providers' text responses (structured outputs are not cached before validation); RAG_GPT_CACHE remains the OAuth cache. Keys are never stored in caches
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

Judge with a separate model and new output (works with both the remote and local architecture):
```fish
"$PY" -m rag_lab.evaluation.provider_judge --judge-provider gemini \
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
No provider switching on 429. Bounded request-level retries preserve the same provider, model and payload; exhausted retries preserve completed rows. Max output is
4096 tokens by default for API providers (override with RAG_LLM_MAX_OUTPUT_TOKENS before starting an experiment); incomplete responses fail rather than being
silently accepted. Provider-specific API and model availability must be tested
with a real key. No live requests are made in unit CI.

## Local checkpoint integration

If your checkout already has rag_lab/core/factory.py and
rag_lab/evaluation/experiment.py, do not replace these or merge older generator
code blindly. Fetch this branch, retrieve the new provider files and the
migration utility, and run the utility with --check first. See its --help.
It applies narrowly scoped changes to the local manifest-bound generator,
preserving the BGE index and all existing experiment identity checks.

New generation rows also record per-question inference_usage (requests, cache hits and token deltas), including optional verification using the shared transport. The provider_judge command binds the exact generation file, rubric and judge inference identity before judging; use a new judge output for a changed generation file. The notebook `notebooks/12_provider_pipeline.ipynb` is a thin entry point for the local BGE corpus experiment.

## Rate-limit handling (adapter v2)

Groq and Gemini API transports now retry the *individual* failed HTTP request,
not the entire question. The OAuth backend is unchanged. Defaults:

| Environment variable | Default | Meaning |
|---|---:|---|
| RAG_LLM_MAX_RETRIES | 4 | Additional attempts per logical request (max 20) |
| RAG_LLM_MAX_RETRY_WAIT | 180 | Cumulative planned sleep budget per logical request, seconds |
| RAG_GROQ_TOKEN_FLOOR | 1000 | Proactive cooldown when reported remaining TPM is at/below this threshold |

HTTP 429 honors numeric or HTTP-date Retry-After plus small positive jitter.
For Groq, zero daily remaining requests respects the daily reset even when
a shorter token reset is present. A long reset exceeding the wait budget
stops immediately; it is never shortened to force a retry. Missing retry
hints use exponential backoff (2, 4, 8, 16 seconds, plus jitter, capped at
30 seconds before jitter). 500/502/503/504 and connection-establishment
timeouts retry within the same budget. Invalid requests, auth/permission
errors, ambiguous read timeouts, incomplete or invalid model responses
are not automatically retried. A read timeout might have incurred usage.

Successful Groq responses with low remaining TPM or zero remaining RPD
schedule a cooldown for the next uncached request. Groq request headers
represent RPD, not RPM; token headers represent TPM. This is a conservative
single-transport controller, not a distributed token bucket. Other clients
and daily token limits can still cause 429s; check the organization's actual
limits at https://console.groq.com/settings/limits. The cooldown is in memory,
not shared across processes or persisted across restarts.

Telemetry: requests counts actual POST attempts (including rejected ones);
successful_responses counts decoded nonempty responses; retry_attempts counts
additional POSTs; rate_limit_events counts HTTP 429s; retry_wait_seconds and
cooldown_wait_seconds report planned sleep seconds; token totals reflect
returned successful usage, not unknown usage from interrupted requests.
Generation and provider_judge rows record the deltas for the completed question
or judge call. A question interrupted after exhausting the policy may still
require replay; its incomplete usage is not represented in completed rows.

Policy settings and adapter version are part of inference identity. The
local manifest also fingerprints source code: use NEW output filenames
after upgrading, even if the provider/model remains identical. Existing
completed generation artifacts may be judged, but use a new judge output.
Do not edit the old manifest to bypass it.

Example (Fish):
```fish
set -gx RAG_LLM_MAX_RETRIES 4
set -gx RAG_LLM_MAX_RETRY_WAIT 180
set -gx RAG_GROQ_TOKEN_FLOOR 1000
```

For a long batch, run a single process per provider/model and avoid launching
other notebook requests simultaneously. A local token-per-day budget or
provider Batch API is a separate scaling option; this policy does not bypass
provider quotas or guarantee a 30-question run fits the daily allowance.
