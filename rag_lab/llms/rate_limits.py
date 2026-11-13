"""Bounded, provider-aware retry and cooldown policy. No network or credentials."""
from __future__ import annotations

import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
POLICY_VERSION = "bounded-header-retry-v1"

@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 4
    max_wait_seconds: float = 180.0
    base_delay_seconds: float = 2.0
    max_backoff_seconds: float = 30.0
    groq_token_floor: int = 1000

    def __post_init__(self):
        if not 0 <= self.max_retries <= 20:
            raise ValueError("max_retries must be between 0 and 20")
        values = (self.max_wait_seconds, self.base_delay_seconds, self.max_backoff_seconds)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Retry delays must be finite and nonnegative")
        if self.base_delay_seconds <= 0 or self.max_backoff_seconds < self.base_delay_seconds:
            raise ValueError("Require 0 < base_delay <= max_backoff")
        if self.groq_token_floor < 0:
            raise ValueError("groq_token_floor must be nonnegative")

    @classmethod
    def from_env(cls):
        return cls(
            max_retries=int(os.getenv("RAG_LLM_MAX_RETRIES", "4")),
            max_wait_seconds=float(os.getenv("RAG_LLM_MAX_RETRY_WAIT", "180")),
            groq_token_floor=int(os.getenv("RAG_GROQ_TOKEN_FLOOR", "1000")),
        )

    def identity(self):
        return {"version": POLICY_VERSION, **asdict(self)}

def headers_lower(headers):
    return {str(k).lower(): str(v).strip() for k, v in (headers or {}).items()}

def seconds(value):
    """Parse nonnegative seconds; reject NaN and infinity."""
    try:
        v = float(value)
        return v if math.isfinite(v) and v >= 0 else None
    except (ValueError, TypeError):
        return None

def retry_after(value, now=None):
    delay = seconds(value)
    if delay is not None:
        return delay
    try:
        stamp = parsedate_to_datetime(value)
        if stamp.tzinfo is None:
            return None
        return max(0.0, stamp.timestamp() - (time.time() if now is None else now))
    except (ValueError, TypeError, OverflowError, AttributeError):
        return None

def reset_seconds(value):
    """Groq duration headers, e.g. 2m59.56s or 7.66s."""
    if value is None:
        return None
    delay = seconds(value)
    if delay is not None:
        return delay
    value = str(value).strip().lower()
    pattern = r"(\d+(?:\.\d+)?)(ms|d|h|m|s)"
    parts = list(re.finditer(pattern, value))
    if not parts or "".join(p.group(0) for p in parts) != value:
        return None
    total = sum(float(p.group(1)) * {"ms": .001, "s": 1, "m": 60, "h": 3600, "d": 86400}[p.group(2)]
                for p in parts)
    return total if math.isfinite(total) else None

def known_cooldown(provider, headers, policy):
    """A conservative preflight delay, not a complete token-bucket predictor."""
    if provider != "groq":
        return 0.0
    h = headers_lower(headers)
    remaining_requests = seconds(h.get("x-ratelimit-remaining-requests"))
    remaining_tokens = seconds(h.get("x-ratelimit-remaining-tokens"))
    delays = [0.0]
    # Groq request headers refer to RPD, not RPM.
    if remaining_requests == 0:
        delays.append(reset_seconds(h.get("x-ratelimit-reset-requests")) or 0.0)
    if remaining_tokens is not None and remaining_tokens <= policy.groq_token_floor:
        delays.append(reset_seconds(h.get("x-ratelimit-reset-tokens")) or 0.0)
    return max(delays)

def retry_delay(provider, status, headers, retry_index, policy, *, jitter=None, now=None):
    """Never shorten a provider-supplied delay; add small positive jitter."""
    h = headers_lower(headers)
    server = retry_after(h.get("retry-after"), now)
    if status == 429 and provider == "groq":
        # Daily exhaustion must not be mistaken for a short TPM reset.
        daily = reset_seconds(h.get("x-ratelimit-reset-requests")) if seconds(h.get("x-ratelimit-remaining-requests")) == 0 else None
        if daily is not None:
            server = max(server or 0.0, daily)
        if server is None:
            server = reset_seconds(h.get("x-ratelimit-reset-tokens"))
    delay = server if server is not None else min(
        policy.max_backoff_seconds, policy.base_delay_seconds * (2 ** retry_index))
    extra = random.uniform(.25, 1.0) if jitter is None else jitter()
    return delay + max(0.0, extra)
