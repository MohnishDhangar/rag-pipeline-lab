"""Versioned, offline-only judge linkage and generation fingerprint.

Safe for reporting and experiments without GPT OAuth, PyJWT, or API keys.
"""
import hashlib
import json

RUBRIC_VERSION = "scientific-judge-v1"


def row_fingerprint(row):
    """Identity of the actual judged question, answer, contexts and sources.

    Keep stable across historical judge outputs. Changes here require
    re-running old judge results under a NEW rubric/fingerprint version.
    """
    content={key:row.get(key) for key in (
        "id","strategy","backend","question",
        "answer","contexts","sources","ground_truth",
        "source_documents","model")}
    data=json.dumps(content,ensure_ascii=False,sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
