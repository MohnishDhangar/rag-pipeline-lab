"""No real HTTP, credentials or sleeps."""
from datetime import datetime, timezone
from email.utils import format_datetime
import pytest
import requests

from rag_lab.llms.providers import APITransport, usage_snapshot, usage_delta, inference_identity
from rag_lab.llms.rate_limits import RetryPolicy, retry_after, reset_seconds
from rag_lab.llms.gpt_adapter import InferenceError

ITEMS = [{"role": "user", "content": "question"}]

class Response:
    def __init__(self, status=200, headers=None):
        self.status_code = status
        self.ok = status == 200
        self.headers = headers or {}
    def json(self):
        return {"choices":[{"finish_reason":"stop","message":{"content":"answer"}}],
                "usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9}}

class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        r = next(self.responses)
        if isinstance(r, Exception):
            raise r
        return r

class Clock:
    def __init__(self):
        self.now = 100.0
        self.waits = []
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds

@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.delenv("RAG_LLM_CACHE", raising=False)

def transport(responses, policy=None):
    clock = Clock()
    session = Session(responses)
    t = APITransport("groq", "test", api_key="fake", http=session,
                     retry_policy=policy or RetryPolicy(), sleep=clock.sleep,
                     clock=clock.monotonic, jitter=lambda: .5)
    return t, clock, session

def test_retry_after_preserves_identical_payload_and_usage():
    t, clock, session = transport([Response(429, {"retry-after":"23"}), Response()])
    before = usage_snapshot(t)
    assert t.generate(ITEMS)[0] == "answer"
    assert clock.waits == [23.5]
    assert session.requests[0] == session.requests[1]
    usage = usage_delta(t, before)
    assert usage["requests"] == 2
    assert usage["successful_responses"] == 1
    assert usage["retry_attempts"] == 1
    assert usage["rate_limit_events"] == 1
    assert usage["retry_wait_seconds"] == 23.5
    assert usage["total_tokens"] == 9

def test_cooldown_from_success_headers_and_daily_header_semantics():
    t, clock, session = transport([Response(headers={
        "x-ratelimit-remaining-requests":"19",
        "x-ratelimit-reset-requests":"23h",
        "x-ratelimit-remaining-tokens":"500",
        "x-ratelimit-reset-tokens":"7.66s"}), Response()])
    t.generate(ITEMS)
    t.generate(ITEMS)
    assert clock.waits == pytest.approx([7.66])
    assert t.cooldown_wait_seconds == pytest.approx(7.66)
    assert t.calls == 2

def test_daily_limit_does_not_sleep_or_repeatedly_request():
    t, clock, session = transport([Response(429, {"Retry-After":"2",
        "x-ratelimit-remaining-requests":"0", "x-ratelimit-reset-requests":"5h"})])
    with pytest.raises(InferenceError, match="429"):
        t.generate(ITEMS)
    assert not clock.waits and t.calls == 1
    with pytest.raises(InferenceError, match="cooldown exceeds"):
        t.generate(ITEMS)
    assert t.calls == 1

def test_retry_budget_is_cumulative_and_never_truncates_server_delay():
    t, clock, session = transport([Response(429, {"Retry-After":"23"}), Response(429, {"Retry-After":"23"})],
                                  RetryPolicy(max_wait_seconds=40))
    with pytest.raises(InferenceError):
        t.generate(ITEMS)
    assert clock.waits == [23.5] and t.calls == 2

def test_5xx_backoff_and_connect_timeout():
    t, clock, session = transport([requests.ConnectTimeout(), Response(503), Response()])
    assert t.generate(ITEMS)[0] == "answer"
    assert clock.waits == [2.5, 4.5]
    assert t.retry_attempts == 2

@pytest.mark.parametrize("failure", [Response(400), Response(401), Response(403),
                                    Response(404), requests.ReadTimeout()])
def test_permanent_or_ambiguous_failure_is_not_retried(failure):
    t, clock, session = transport([failure])
    with pytest.raises(InferenceError):
        t.generate(ITEMS)
    assert t.calls == 1 and not clock.waits

def test_max_attempts():
    t, clock, session = transport([Response(429)] * 3, RetryPolicy(max_retries=2))
    with pytest.raises(InferenceError):
        t.generate(ITEMS)
    assert t.calls == 3 and clock.waits == [2.5,4.5]

def test_header_parsing():
    assert reset_seconds("2m59.56s") == pytest.approx(179.56)
    assert reset_seconds("1h30m") == 5400
    assert reset_seconds("20ms") == .02
    assert reset_seconds("NaN") is None
    date = format_datetime(datetime.fromtimestamp(120, timezone.utc), usegmt=True)
    assert retry_after(date, now=100) == 20
    assert retry_after("23") == 23

def test_policy_fingerprinted(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MAX_RETRIES","3")
    a = inference_identity("groq","test")
    monkeypatch.setenv("RAG_LLM_MAX_RETRIES","4")
    b = inference_identity("groq","test")
    assert a != b
    assert b["retry_policy"]["max_retries"] == 4
