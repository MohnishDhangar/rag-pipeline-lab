"""Explicit, interchangeable inference providers; never fall back across providers."""
from __future__ import annotations

import copy
import logging
import time
import json
import os
import re
from typing import Any

import requests
from pydantic import PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from rag_lab.core.cache import ResponseCache
from .rate_limits import (RetryPolicy, RETRYABLE_STATUS, known_cooldown,
                          retry_delay, headers_lower, retry_after)
from .gpt_adapter import (
    GPTTransport, InferenceError, _convert, build_chat_model as oauth_model, parse_object,
)

PROVIDERS = ("chatgpt_oauth", "groq", "gemini")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
MODEL_ENV = {"chatgpt_oauth": "RAG_GPT_MODEL", "groq": "RAG_GROQ_MODEL",
             "gemini": "RAG_GEMINI_MODEL"}
ADAPTER_VERSION = "provider-adapter-v2"
LOG = logging.getLogger(__name__)
MAX_OUTPUT_TOKENS = 4096

def output_budget():
    value = int(os.getenv("RAG_LLM_MAX_OUTPUT_TOKENS", str(MAX_OUTPUT_TOKENS)))
    if value < 1:
        raise ValueError("RAG_LLM_MAX_OUTPUT_TOKENS must be positive")
    return value

def resolve_provider(provider=None):
    selected = provider or os.getenv("RAG_LLM_PROVIDER") or "chatgpt_oauth"
    if selected not in PROVIDERS:
        raise ValueError(f"Unknown inference provider: {selected}")
    return selected

def resolve_model(provider, model=None):
    model = model or os.getenv(MODEL_ENV[provider])
    if not model and provider == "groq":
        model = DEFAULT_GROQ_MODEL
    if not model:
        raise ValueError(f"Set {MODEL_ENV[provider]} or specify an explicit model")
    if provider == "gemini":
        model = model.removeprefix("models/")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
            raise ValueError("Invalid Gemini model identifier")
    return model

def inference_identity(provider, model):
    return {"provider": provider, "model": model,
            "adapter_version": ADAPTER_VERSION,
            "max_output_tokens": None if provider == "chatgpt_oauth" else output_budget(),
            "structured_output": "prompt-json" if provider == "chatgpt_oauth" else "provider-json-schema",
            "cache_enabled": bool(os.getenv("RAG_GPT_CACHE" if provider == "chatgpt_oauth" else "RAG_LLM_CACHE")),
            **({"retry_policy": RetryPolicy.from_env().identity()} if provider != "chatgpt_oauth" else {})}

def _schema(value):
    return value.model_json_schema() if hasattr(value, "model_json_schema") else copy.deepcopy(value)

class APITransport:
    """Non-streaming API transport with bounded retries; never changes model."""
    def __init__(self, provider, model, *, http=None, api_key=None, retry_policy=None,
                 sleep=None, clock=None, jitter=None):
        if provider not in ("groq", "gemini"):
            raise ValueError("APITransport requires groq or gemini")
        self.provider, self.model = provider, resolve_model(provider, model)
        self.max_output_tokens = output_budget()
        self.http = http or requests.Session()
        self.api_key = api_key or (os.getenv("GROQ_API_KEY") if provider == "groq"
                                  else os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if not self.api_key:
            raise ValueError("Set GROQ_API_KEY" if provider == "groq" else "Set GEMINI_API_KEY or GOOGLE_API_KEY")
        self.calls = 0  # all actual HTTP attempts, including rejected requests
        self.successful_responses = 0
        self.retry_attempts = 0
        self.rate_limit_events = 0
        self.retry_wait_seconds = 0.0
        self.cooldown_wait_seconds = 0.0
        self.retry_policy = retry_policy or RetryPolicy.from_env()
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._jitter = jitter
        self._cooldown_until = 0.0
        self.cache_hits = 0
        self.usage_totals = dict(input_tokens=0, output_tokens=0, total_tokens=0)
        self.cache = ResponseCache(os.environ["RAG_LLM_CACHE"]) if os.getenv("RAG_LLM_CACHE") else None

    def _request(self, items, instructions, schema):
        if self.provider == "groq":
            messages = ([{"role": "system", "content": instructions}] if instructions else []) + list(items)
            payload = {"model": self.model, "messages": messages, "stream": False,
                       "max_completion_tokens": self.max_output_tokens}
            if schema is not None:
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "rag_result", "strict": False, "schema": schema}}
            return ("https://api.groq.com/openai/v1/chat/completions",
                    {"Authorization": "Bearer " + self.api_key}, payload)
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in items]
        config = {"maxOutputTokens": self.max_output_tokens}
        if schema is not None:
            config.update(responseMimeType="application/json", responseJsonSchema=schema)
        payload = {"contents": contents, "generationConfig": config}
        if instructions:
            payload["systemInstruction"] = {"parts": [{"text": instructions}]}
        return (f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                {"x-goog-api-key": self.api_key}, payload)

    def _decode(self, data):
        if self.provider == "groq":
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise InferenceError("Groq response incomplete", code="incomplete_response")
            text = choice["message"]["content"]
            raw = data.get("usage") or {}
            usage = {"input_tokens": raw.get("prompt_tokens", 0),
                     "output_tokens": raw.get("completion_tokens", 0),
                     "total_tokens": raw.get("total_tokens", 0)}
        else:
            candidate = data["candidates"][0]
            if candidate.get("finishReason") != "STOP":
                raise InferenceError("Gemini response incomplete or blocked", code="incomplete_response")
            text = "".join(p.get("text", "") for p in candidate["content"]["parts"]
                           if not p.get("thought"))
            raw = data.get("usageMetadata") or {}
            usage = {"input_tokens": raw.get("promptTokenCount", 0),
                     "output_tokens": raw.get("candidatesTokenCount", 0) + raw.get("thoughtsTokenCount", 0),
                     "total_tokens": raw.get("totalTokenCount", 0)}
        if not isinstance(text, str) or not text.strip():
            raise InferenceError(f"{self.provider} returned no answer text", code="empty_response")
        return text, usage

    def _post_with_retry(self, url, headers, payload):
        policy = self.retry_policy
        spent = 0.0
        cooldown = max(0.0, self._cooldown_until - self._clock())
        if cooldown:
            if cooldown > policy.max_wait_seconds:
                raise InferenceError(
                    f"{self.provider} known rate-limit cooldown exceeds wait budget",
                    code="rate_limit", status=429)
            LOG.warning("%s cooldown: waiting %.2fs before next request", self.provider, cooldown)
            self._sleep(cooldown)
            self.cooldown_wait_seconds += cooldown
            spent += cooldown
        for attempt in range(policy.max_retries + 1):
            self.calls += 1
            if attempt:
                self.retry_attempts += 1
            response_headers = {}
            try:
                response = self.http.post(
                    url, headers={**headers, "Content-Type": "application/json"},
                    json=payload, timeout=(20, 240))
            except requests.ConnectTimeout:
                # Requests documents ConnectTimeout as safe to retry.
                status = None
                error = InferenceError(f"{self.provider} connection timed out",
                                       code="transport_error")
            except requests.RequestException as exc:
                # A read timeout/disconnect may have incurred billable inference.
                # Do not silently duplicate an ambiguous POST.
                raise InferenceError(f"{self.provider} transport interrupted: {type(exc).__name__}",
                                     code="transport_error") from None
            else:
                response_headers = headers_lower(response.headers)
                if response.ok:
                    self._cooldown_until = self._clock() + known_cooldown(
                        self.provider, response_headers, policy)
                    return response
                status = response.status_code
                if status == 429:
                    self.rate_limit_events += 1
                code = {401: "authentication", 403: "permission", 429: "rate_limit"}.get(status, "http_error")
                hint = retry_after(response_headers.get("retry-after"))
                suffix = f"; Retry-After={hint:g}" if hint is not None else ""
                error = InferenceError(f"{self.provider} HTTP {status}{suffix}",
                                       code=code, status=status)
                if status not in RETRYABLE_STATUS:
                    raise error
            delay = retry_delay(self.provider, status, response_headers, attempt,
                                policy, jitter=self._jitter)
            # Keep a known 429 cooldown even when stopping after the final attempt.
            if status == 429:
                self._cooldown_until = max(self._cooldown_until, self._clock() + delay)
            if attempt >= policy.max_retries or spent + delay > policy.max_wait_seconds:
                LOG.warning("%s retry budget exhausted (attempts=%d, waited=%.2fs, next=%.2fs)",
                            self.provider, attempt + 1, spent, delay)
                raise error
            LOG.warning("%s %s: retry %d/%d in %.2fs",
                        self.provider, status if status is not None else "connect timeout",
                        attempt + 1, policy.max_retries, delay)
            self._sleep(delay)
            self.retry_wait_seconds += delay
            spent += delay
        raise AssertionError("unreachable")

    def generate(self, input, instructions=None, *, schema=None):
        url, headers, payload = self._request(input, instructions, schema)
        # Do not cache unvalidated structured output. A schema-invalid response
        # must never poison subsequent grading retries.
        key = self.cache.key(self.provider + ":" + self.model, payload) if self.cache and schema is None else None
        if key:
            cached = self.cache.get(key)
            if cached:
                self.cache_hits += 1
                return cached[0], {**cached[1], "cached": True}
        response = self._post_with_retry(url, headers, payload)
        try:
            text, usage = self._decode(response.json())
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InferenceError(f"{self.provider} invalid response", code="invalid_response") from None
        self.successful_responses += 1
        for field in self.usage_totals:
            self.usage_totals[field] += usage[field]
        if key:
            self.cache.put(key, text, usage)
        return text, usage

    def generate_structured(self, input, instructions, schema):
        return self.generate(input, instructions, schema=_schema(schema))

class ProviderChatModel(BaseChatModel):
    model_name: str
    provider: str
    _transport: Any = PrivateAttr()

    def __init__(self, *, transport, **kwargs):
        super().__init__(**kwargs)
        self._transport = transport

    @property
    def _llm_type(self):
        return self.provider

    @property
    def _identifying_params(self):
        return {"provider": self.provider, "model": self.model_name}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop is not None or kwargs:
            raise ValueError("Unsupported provider generation options")
        items, instructions = _convert(messages)
        text, usage = self._transport.generate(items, instructions=instructions)
        meta = {k: usage.get(k, 0) for k in ("input_tokens", "output_tokens", "total_tokens")}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=meta))])

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        if kwargs:
            raise ValueError("Unsupported structured output options")
        if not hasattr(schema, "model_validate"):
            raise TypeError("Use a Pydantic schema for validated structured output")
        def structured(value):
            messages = convert_to_messages(value.to_messages() if hasattr(value, "to_messages") else value)
            items, instructions = _convert(messages)
            raw, usage = self._transport.generate_structured(items, instructions, schema)
            msg = AIMessage(content=raw)
            try:
                parsed = schema.model_validate_json(raw)
            except ValueError as exc:
                if include_raw:
                    return {"raw": msg, "parsed": None, "parsing_error": exc}
                raise InferenceError("Provider output failed schema validation", code="invalid_structured_output") from None
            return {"raw": msg, "parsed": parsed, "parsing_error": None} if include_raw else parsed
        return RunnableLambda(structured)

def build_transport(provider=None, model=None, **kwargs):
    provider = resolve_provider(provider)
    model = resolve_model(provider, model)
    if provider == "chatgpt_oauth":
        return GPTTransport(model=model, **kwargs)
    return APITransport(provider, model, **kwargs)

def build_chat_model(model=None, transport=None, *, provider=None):
    provider = resolve_provider(provider)
    model = resolve_model(provider, model)
    if provider == "chatgpt_oauth":
        return oauth_model(model=model, transport=transport)
    return ProviderChatModel(provider=provider, model_name=model,
                             transport=transport or build_transport(provider, model))

def judge_selection(provider=None, model=None):
    chosen = resolve_provider(provider or os.getenv("RAG_JUDGE_PROVIDER"))
    return chosen, resolve_model(chosen, model or os.getenv("RAG_JUDGE_MODEL"))

def usage_snapshot(llm):
    transport = getattr(llm, "_transport", llm)
    totals = getattr(transport, "usage_totals", {}) or {}
    return {"requests": getattr(transport, "calls", 0),
            "cache_hits": getattr(transport, "cache_hits", 0),
            **{k: getattr(transport, k, 0) for k in ("successful_responses", "retry_attempts",
                 "rate_limit_events", "retry_wait_seconds", "cooldown_wait_seconds")},
            **{k: totals.get(k, 0) for k in ("input_tokens", "output_tokens", "total_tokens")}}

def usage_delta(llm, before):
    after = usage_snapshot(llm)
    return {k: after[k] - before.get(k, 0) for k in after}
