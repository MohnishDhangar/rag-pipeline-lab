import json
import pytest

from rag_lab.knowledge.paper_registry import PaperRegistry
from rag_lab.knowledge.citations import format_context, citation_coverage
from rag_lab.core.trace import source_record
from rag_lab.evaluation.metrics import score_retrieval
from rag_lab.evaluation.provenance_repair import repair_row

DATA={"metadata":{"papers":[
    {"short_id":"ntk_pinn","title":"When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective","file":None},
    {"short_id":"sk_pinn","title":"SK-PINN: Accelerated physics-informed deep learning by smoothing kernel gradients","file":None},
]}}
class Doc:
    def __init__(self,body,meta):self.page_content,self.metadata=body,meta


def test_registry_resolves_known_filename_alias_and_rejects_unknown():
    reg=PaperRegistry.from_eval(DATA)
    assert reg.resolve("/home/a/When and Why PINNs fail to train: A Neural Network Perspective.pdf")=="ntk_pinn"
    assert reg.resolve("SK-PINN_ Accelerated physics-informed deep learning by smoothing kernel gradients.pdf")=="sk_pinn"
    assert reg.resolve("unknown.pdf") is None


def test_passage_ids_unique_and_pdf_pages_one_based():
    reg=PaperRegistry.from_eval(DATA)
    meta={"source":"/home/a/When and Why PINNs fail to train: A Neural Network Perspective.pdf","page":7}
    d1=Doc("first passage",reg.enrich(meta,"first passage"))
    d2=Doc("second passage",reg.enrich(meta,"second passage"))
    assert d1.metadata["id"] != d2.metadata["id"]
    assert d1.metadata["pdf_page_number"]==8
    assert d1.metadata["paper_id"]=="ntk_pinn"
    record=source_record(d1)
    tag="[SOURCE "+record["id"]+":pdfpage8]"
    assert tag in format_context(d1)
    assert citation_coverage("An explanation "+tag,[record])["validity"]==1.0
    assert citation_coverage("Wrong "+tag,[source_record(d2)])["validity"]==0.0


def test_legacy_repair_without_changing_generated_answer():
    reg=PaperRegistry.from_eval(DATA)
    old_id="When and Why PINNs fail to train: A Neural Network Perspective.pdf"
    answer="Correct [SOURCE "+old_id+":page7]"
    row={"id":"q","answer":answer,"contexts":["PDF evidence"],
         "sources":[{"id":old_id,"page":7,"source":"/home/a/"+old_id}]}
    result=repair_row(row,reg)
    assert result["answer"]==answer
    assert result["sources"][0]["paper_id"]=="ntk_pinn"
    assert result["sources"][0]["legacy_id"]==old_id
    assert citation_coverage(answer,result["sources"])["validity"]==1.0


def test_chunk_precision_and_paper_recall_do_not_deduplicate_chunk_rankings():
    scores=score_retrieval(["ntk_pinn","sk_pinn","ntk_pinn",None],["ntk_pinn"],k=4)
    assert scores["provenance_coverage"]==.75
    assert scores["precision_at_k"] is None # unknown provenance, not a false negative
    s=score_retrieval(["ntk_pinn","sk_pinn","ntk_pinn","sk_pinn"],["ntk_pinn"],k=4)
    assert s["chunk_precision_at_k"]==.5
    assert s["paper_recall_at_k"]==1.0
    assert s["mrr_at_k"]==1.0
    assert 0 <= s["ndcg_at_k"] <= 1.0


def test_registry_never_guesses_similar_papers():
    reg=PaperRegistry.from_eval(DATA)
    assert reg.resolve("A Neural Tangent Kernel Study of PINN Failure.pdf") is None
    with pytest.raises(ValueError,match="Conflicting"):
        reg.enrich({"source":"When and Why PINNs fail to train: A Neural Network Perspective.pdf",
                    "paper_id":"sk_pinn"},"text")
