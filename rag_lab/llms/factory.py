"""Stable public inference factory for notebooks, CLI and serving."""
from .providers import (
    PROVIDERS, build_chat_model, build_transport, inference_identity,
    resolve_provider, resolve_model, judge_selection, usage_snapshot, usage_delta,
)
