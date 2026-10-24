"""Canonical scientific paper registry: no question-label leakage or fuzzy guesses."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path

# Audited filename aliases only. Null 'file' fields in eval_set are otherwise
# left unresolved until explicitly mapped. Never use a question's gold source.
ALIASES = {
    "ntk_pinn": ["When and Why PINNs fail to train: A Neural Network Perspective.pdf"],
    "sk_pinn": ["SK-PINN_ Accelerated physics-informed deep learning by smoothing kernel gradients.pdf"],
    # Verified against stored Chroma metadata (line breaks are normalized by file_key).
    "orig_pinn": [
        "PINN for solving forward and inverse problems involving nonlinear partial differential equations.pdf"
    ],
    "xpinn": [
        "XPINNs - A Generalized Space-Time Domain Decomposition "
        "Based Deep Learning Framework for Nonlinear "
        "Partial Differential Equations.pdf"
    ],
}


def file_key(value: str) -> str:
    stem = Path(str(value).replace("\\", "/")).name
    return re.sub(r"[^a-z0-9]+", "", stem.removesuffix(".pdf").lower())


def passage_id(paper_id, source, page_index, text):
    identity = (str(paper_id or "unknown")+"|"+file_key(source or "unknown")
                +"|"+str(page_index)+"|"+text)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9_-]+","",str(paper_id or "unknown").lower())
    return slug+":"+digest


class PaperRegistry:
    def __init__(self, papers):
        self.by_filename = {}
        for paper in papers:
            name = paper["short_id"]
            matches = [paper.get("file"), paper.get("title"), *ALIASES.get(name, [])]
            for label in filter(None,matches):
                key=file_key(label)
                original=self.by_filename.get(key)
                if original and original != name:
                    raise ValueError(f"Ambiguous paper alias {label!r}: {original} vs {name}")
                self.by_filename[key]=name

    @classmethod
    def from_eval(cls, dataset):
        return cls(dataset["metadata"]["papers"])

    def resolve(self, source):
        if not source: return None
        return self.by_filename.get(file_key(source))

    def enrich(self, metadata, content):
        data=dict(metadata or {})
        source=data.get("source")
        resolved=self.resolve(source)
        claimed=data.get("paper_id") or data.get("short_id")
        if claimed and resolved and claimed != resolved:
            raise ValueError("Conflicting stored paper_id and canonical registry mapping")
        if resolved:
            data["paper_id"]=resolved
        elif claimed:
            data["paper_id"]=claimed
        if "page" in data and data["page"] is not None:
            page=int(data["page"])
            data["pdf_page_index"]=page
            data["pdf_page_number"]=page+1
        if source or content:
            # Preserve original provider/index ID for diagnostics.
            if "id" in data:
                data.setdefault("original_id",data["id"])
            data["id"]=passage_id(data.get("paper_id"),source,data.get("pdf_page_index"),content)
        return data
