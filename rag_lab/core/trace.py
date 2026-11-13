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
    meta = dict(getattr(document, "metadata", {}) or {})
    text = getattr(document, "page_content", "")
    source = meta.get("source")
    default_id = (str(source).replace("\\", "/").split("/")[-1]
                  if source else hashlib.sha256(text.encode()).hexdigest()[:16])
    page_index = meta.get("pdf_page_index", meta.get("page"))
    page_number = meta.get("pdf_page_number")
    # PDF loader pages are ZERO-BASED; converted PDF pages are ONE-BASED.
    if page_number is None and page_index is not None:
        page_number = int(page_index) + 1
    return {
        "id": str(meta.get("id") or default_id),
        "paper_id": meta.get("paper_id") or meta.get("short_id"),
        "source": source,
        "page": meta.get("page"),  # legacy zero-based indexing
        "pdf_page_index": page_index,
        "pdf_page_number": page_number,
        "score": meta.get("score"),
        "original_id": meta.get("original_id"),
        "legacy_id": meta.get("legacy_id"),
    }


class TraceableStrategy:
    """Strategies set self.last_contexts during run, never in a second retrieval."""
    def run_with_trace(self, question):
        self.last_contexts = []
        transport=getattr(getattr(self,"llm",None),"_transport",None)
        before=getattr(transport,"calls",0)
        start = time.perf_counter()
        answer = self.run(question)
        docs = self.last_contexts
        return RAGResult(question=question, answer=answer,
                         strategy=type(self).__name__,
                         contexts=[getattr(d,"page_content",str(d)) for d in docs],
                         sources=[source_record(d) for d in docs],
                         latency_ms=round(1000*(time.perf_counter()-start),2),
                         trace={"gpt_requests":getattr(transport,"calls",0)-before})
