"""Shared serving/evaluation source provenance boundary.

Never construct a serving strategy from a raw vector store: canonicalize
passage IDs, paper IDs and page metadata before any model sees evidence.
This does not consult gold labels on individual evaluation questions.
"""
from __future__ import annotations
import json
from pathlib import Path
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.retrieval.provenance import ProvenanceStore


def serving_store(raw_store, *, registry_path):
    dataset=json.loads(Path(registry_path).read_text(encoding="utf-8"))
    registry=PaperRegistry.from_eval(dataset)
    return ProvenanceStore(raw_store,registry),registry
