"""Same-pass answer+evidence tracing; no additional retrieval or GPT call."""
from __future__ import annotations
import hashlib, time
from dataclasses import asdict, dataclass, field

@dataclass
class RAGResult:
    question: str
    answer: str
    strategy: str
    contexts: list[str]
    sources: list[dict]
    latency_ms: float
    trace: dict = field(default_factory=dict)

    def to_dict(self): return asdict(self)

def source_record(document):
    meta = dict(getattr(document,"metadata",{}) or {})
    text = getattr(document,"page_content","")
    return {
        "id": str(meta.get("id") or hashlib.sha256(text.encode()).hexdigest()[:16]),
        "paper_id": meta.get("paper_id") or meta.get("short_id"),
        "source": meta.get("source"),
        "page": meta.get("page"),
        "score": meta.get("score"),
    }

class TraceableStrategy:
    """Strategies set self.last_contexts during run, never in a second retrieval."""
    def run_with_trace(self, question):
        self.last_contexts = []
        start = time.perf_counter()
        answer = self.run(question)
        docs = self.last_contexts
        return RAGResult(question=question, answer=answer,
                         strategy=type(self).__name__,
                         contexts=[getattr(d,"page_content",str(d)) for d in docs],
                         sources=[source_record(d) for d in docs],
                         latency_ms=round(1000*(time.perf_counter()-start),2))
