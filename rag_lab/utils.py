"""
Shared utilities used across every RAG-Lab notebook: retry/backoff with
daily-quota detection, JSONL checkpointing, and per-strategy API-call
pacing.

This is the SAME utils.py from the generation/scoring phase, with one
addition: `call_with_backoff_async`, the async counterpart used by the
native-`await` RAGAS scoring loop (09/10 use it; the sync version is still
used by the generation loops since LangChain's `.invoke()` is sync).

Drop this in at rag_lab/utils.py (it replaces the existing file — nothing
here was removed, only added).
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# Quota / retry
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Replaces smart-Unicode punctuation with plain ASCII.

    Kept as a cheap defensive measure against a class of Gemini 500s we saw
    mid-project. Never fully proven to be THE root cause on its own (a
    kernel restart made the errors stop recurring before this could be
    isolated in a controlled test) -- but it's free insurance, so it stays.
    """
    if not isinstance(text, str):
        return text
    replacements = {
        "‘": "'", "’": "'",
        "“": '"', "”": '"',
        "–": "-", "—": "-",
        "…": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def is_daily_quota_error(e: Exception) -> bool:
    """True for a hard daily-quota wall (TPD) -- NOT a transient rate limit.

    Distinguishing these matters: a transient 429/500 is worth retrying
    with backoff, but a TPD wall won't clear for the stated cooldown, so
    retrying it just burns more wall-clock time for nothing. Both Groq's
    "tokens per day (TPD)" message and Google's RESOURCE_EXHAUSTED daily
    cap land here.
    """
    msg = str(e)
    return (
        "tokens per day" in msg
        or "TPD" in msg
        or "RequestsPerDayPerProjectPerModel" in msg
        or "GenerateRequestsPerDayPerProjectPerModel" in msg
    )


def call_with_backoff(
    fn: Callable,
    max_retries: int = 4,
    pre_delay: float = 2.0,
    retryable_markers: Iterable[str] = ("500", "INTERNAL", "429", "RESOURCE_EXHAUSTED"),
):
    """Calls fn() with exponential backoff on transient errors.

    Raises immediately (no retries wasted) on:
      - a detected daily-quota wall (is_daily_quota_error)
      - any error that doesn't match retryable_markers

    Backs off exponentially (pre_delay * 2**attempt, roughly) on anything
    that does match, up to max_retries attempts.
    """
    delay = pre_delay
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if is_daily_quota_error(e):
                raise
            msg = str(e)
            if not any(marker in msg for marker in retryable_markers):
                raise
            if attempt == max_retries:
                raise
            print(f"Transient error, retrying in {delay:.1f}s (attempt {attempt}/{max_retries}): {msg[:200]}")
            time.sleep(delay)
            delay *= 1.8
    raise last_exc


async def call_with_backoff_async(
    fn: Callable,
    max_retries: int = 4,
    pre_delay: float = 2.0,
    retryable_markers: Iterable[str] = ("500", "INTERNAL", "429", "RESOURCE_EXHAUSTED"),
):
    """Async counterpart of call_with_backoff, for use inside an
    `async def` loop driven by Jupyter's native top-level `await`
    (NOT asyncio.run(), NOT nest_asyncio -- both were confirmed broken
    inside this project's Python 3.14 + already-running-event-loop setup).

    `fn` may be sync or async: if `fn()` returns a coroutine, it's awaited;
    otherwise the (sync) result is returned as-is. This lets the same
    helper wrap both a plain `.invoke()` call and a RAGAS `.ascore()` call.
    """
    delay = pre_delay
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            last_exc = e
            if is_daily_quota_error(e):
                raise
            msg = str(e)
            if not any(marker in msg for marker in retryable_markers):
                raise
            if attempt == max_retries:
                raise
            print(f"Transient error, retrying in {delay:.1f}s (attempt {attempt}/{max_retries}): {msg[:200]}")
            await asyncio.sleep(delay)
            delay *= 1.8
    raise last_exc


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_completed_keys(
    jsonl_path,
    key_fields=("strategy", "id"),
    require_fields: Optional[Iterable[str]] = None,
) -> set:
    """Scans a checkpoint JSONL and returns the set of (key_fields...) tuples
    already present, so a resumed loop can skip them.

    `require_fields`, when given, lists fields that must be non-None for a
    row to count as "done" -- this is what stops a corrupted/partial row
    (e.g. one written mid-crash, or one where every RAGAS metric came back
    NaN) from being silently treated as complete. A row failing that check
    is NOT added to the returned set, so the next run will retry it and the
    retry's fresh row supersedes the bad one once re-deduplicated on read.
    """
    jsonl_path = Path(jsonl_path)
    completed = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if require_fields and any(row.get(field) is None for field in require_fields):
                continue
            try:
                key = tuple(row[field] for field in key_fields)
            except KeyError:
                continue
            completed.add(key)
    return completed


def append_jsonl(jsonl_path, record: dict):
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------

# Rough number of LLM calls each strategy makes per question -- used to
# pace requests under Groq's free-tier RPM/TPD ceilings. decomposition is
# the expensive one: it generates sub-questions, answers each, then
# synthesizes.
STRATEGY_CALL_COUNTS = {
    "naive": 1,
    "multi_query": 2,
    "rag_fusion": 2,
    "hyde": 2,
    "step_back": 2,
    "decomposition": 5,
    "crag": 3,  # retrieve-grade (1 call/doc, batched to 1) + refine (0-1) + generate (1); ~3 typical
}


def pacing_delay(strategy_name: str, base_delay: float = 2.0) -> float:
    return base_delay * STRATEGY_CALL_COUNTS.get(strategy_name, 2)
