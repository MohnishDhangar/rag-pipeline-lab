"""The v4 CRAG trace records passage-level grader decisions and evidence scope."""
from langchain_core.documents import Document
from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.strategies.crag import CRAGStrategy,CRAG_VERSION


def test_grade_trace_distinguishes_paper_presence_from_accepted_evidence():
    registry=PaperRegistry([{"short_id":"orig_pinn","file":"orig.pdf"}])
    class Storage:
        pass
    class FakeLLM:
        pass
    strategy=CRAGStrategy(Storage(),FakeLLM(),registry=registry,
                          grader_fn=lambda q,docs:[])
    docs=[Document(page_content="The authors used L-BFGS.",
                   metadata={"source":"orig.pdf","page":5})]
    grades=[type("Grade",(),{
        "grade":"ambiguous","quote":"","reason":"Only generic optimizer name"})()]
    trace=strategy._grade_trace(docs,grades)
    assert CRAG_VERSION=="closed-corpus-crag-v5"
    assert trace[0]["paper_id"]=="orig_pinn"
    assert trace[0]["grade"]=="ambiguous"
    assert not trace[0]["quote_verified"]
