"""JSONL checkpoint reader with no model, network or OAuth dependencies."""
import json
from pathlib import Path


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Match the legacy checkpoint reader, which tolerates
                # a partially written record at the end of an interrupted run.
                continue
    return rows
