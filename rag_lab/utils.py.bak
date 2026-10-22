"""
Shared utilities for rag_lab. Every fix here is informed by concrete
failures hit during development — see comments on each function for
what specifically broke and why this is the mitigation.
"""

import json
import random
import time
from pathlib import Path

_UNICODE_REPLACEMENTS = {
    "\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2014": "--",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
}


def normalize_text(text: str) -> str:
    """Replace 'smart' typographic Unicode with plain ASCII. Kept as a
    cheap, harmless defensive measure. NOTE: during development this
    was initially suspected as the sole cause of intermittent Gemini
    embedding 500 errors, but later testing showed a confirmed-clean
    ASCII string could still fail, and a kernel restart also appeared
    to resolve things independently — root cause was never fully
    isolated. If 500s recur on clean text, restart the kernel before
    assuming this function is insufficient."""
    for bad, good in _UNICODE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def is_daily_quota_error(e: Exception) -> bool:
    """True for a hard daily-token-cap (TPD) error from Groq or Gemini.
    Unlike a per-minute rate limit, this will not clear on the
    timescale of exponential backoff — retrying wastes real time.
    Callers should stop the whole run on this, not just skip a retry."""
    msg = str(e)
    return "tokens per day" in msg or "TPD" in msg


def call_with_backoff(
    fn,
    max_retries: int = 4,
    pre_delay: float = 2.0,
    retryable_markers: tuple = ("500", "INTERNAL", "429", "RESOURCE_EXHAUSTED"),
):
    """Wraps a zero-arg call with a pre-call delay and exponential
    backoff for genuinely transient errors. A daily-quota (TPD) error
    is detected and re-raised IMMEDIATELY, consuming no retry — Groq's
    TPD message contains '429' and would otherwise match the generic
    retryable pattern and waste ~15s per call thrashing against a wall
    that won't open for minutes or hours. Anything not matching a known
    retryable pattern is also re-raised immediately, so real bugs are
    never silently retried away."""
    time.sleep(pre_delay)
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if is_daily_quota_error(e):
                raise
            msg = str(e)
            if not any(marker in msg for marker in retryable_markers):
                raise
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) * 2 + random.uniform(0, 1)
            print(f"  Transient error, retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries}): {msg[:100]}")
            time.sleep(wait)


def load_completed_keys(jsonl_path: Path, key_fields=("strategy", "id"), require_fields=None) -> set:
    """Generic JSONL checkpoint reader — returns composite keys already
    done. If require_fields is given, a record only counts as done when
    ALL those fields are non-None. Without this, a row corrupted by a
    systemic failure (e.g. every metric scored None because RAGAS's
    Executor silently failed) gets marked complete forever and is never
    retried. With it, such rows are automatically re-attempted."""
    completed = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if require_fields and not all(rec.get(field) is not None for field in require_fields):
                continue
            completed.add(tuple(rec[k] for k in key_fields))
    return completed


def append_jsonl(jsonl_path: Path, record: dict):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


STRATEGY_CALL_COUNTS = {
    "naive": 1, "multi_query": 2, "rag_fusion": 2,
    "hyde": 2, "step_back": 2, "decomposition": 5,
}


def pacing_delay(strategy_name: str, base_delay: float = 2.0) -> float:
    """Paces proportionally to a strategy's actual LLM-call cost —
    decomposition (5 calls/query) gets more breathing room than naive
    (1 call/query), rather than one flat delay for every strategy."""
    return base_delay * STRATEGY_CALL_COUNTS.get(strategy_name, 2)