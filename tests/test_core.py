import math
from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.core.trace import TraceableStrategy
from rag_lab.core.reliability import assess
from rag_lab.knowledge.evidence_graph import EvidenceGraph
from rag_lab.knowledge.citations import citation_coverage, format_context

class Doc:
    def __init__(self,text,**metadata):
        self.page_content=text
        self.metadata=metadata

class DummyStrategy(TraceableStrategy):
    def run(self,q):
        self.last_contexts=[Doc("Evidence about PINNs",id="chunk-1",paper_id="pinn",page=3)]
        return "A supported finding [SOURCE chunk-1:pdfpage4]"

def test_trace_is_from_generation_not_second_retrieval():
    obj=DummyStrategy()
    result=obj.run_with_trace("What?")
    assert result.contexts==["Evidence about PINNs"]
    assert result.sources[0]["page"]==3
    assert result.latency_ms>=0
    assert assess(result,require_citations=True).action=="accept"
    assert assess(result,cross_document=True).action=="verify"

def test_citations():
    d=Doc("Text",id="x",page=9)
    assert "[SOURCE x:pdfpage10]" in format_context(d)
    assert citation_coverage("Claim [SOURCE x:pdfpage10]",[{"id":"x","page":9,"pdf_page_number":10}])["validity"]==1.0
    assert citation_coverage("Claim [SOURCE wrong]",[{"id":"x"}])["validity"]==0.0

def test_metrics_and_unknown_labels():
    m=score_retrieval(["A","A","B","C"],["A","B"],k=4)
    assert m["recall_at_k"]==1.0
    assert m["mrr_at_k"]==1.0
    assert math.isclose(m["precision_at_k"],0.75)
    assert score_retrieval([],[],k=4)["recall_at_k"] is None

def test_evidence_graph_roundtrip(tmp_path):
    graph=EvidenceGraph()
    graph.add_paper("p","Example")
    graph.add_passage("p","Physics-informed neural networks solve PDEs using residual losses",2,"a.pdf")
    path=tmp_path/"graph.json"
    graph.save(path)
    loaded=EvidenceGraph.load(path)
    hits=loaded.query("Which PINN method solves PDEs?",k=4)
    assert hits and hits[0].paper_id=="p"
