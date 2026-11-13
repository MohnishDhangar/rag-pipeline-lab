"""All tests are offline; no model keys required."""
from langchain_core.documents import Document
from rag_lab.retrieval.query_router import classify_question
from rag_lab.retrieval.adaptive_store import AdaptiveRetrievalStore


class FakeRetriever:
    def __init__(self, owner, mode):
        self.owner, self.mode = owner, mode

    def invoke(self, question, config=None, **kwargs):
        self.owner.queries.append((self.mode, question))
        return [Document(page_content="evidence", metadata={"source":"paper.pdf", "page":0})]


class FakeDense:
    def __init__(self):
        self.queries = []

    def as_retriever(self, search_type="similarity", search_kwargs=None):
        return FakeRetriever(self, search_type)


def test_router_selects_cross_without_gold_labels():
    examples = [
        "Both XPINN and SK-PINN aim to reduce PINN training cost. Contrast their strategies.",
        "The PINN-GNN PUE paper and the XPINN paper both use decomposition; contrast how each works.",
        "GNS uses learned message passing while PINNs enforce PDE residuals; what is the trade-off?",
        "All three papers in this batch propose ways to make training efficient.",
    ]
    assert all(classify_question(q).search_type == "mmr" for q in examples)


def test_router_does_not_confuse_two_scenarios_with_two_papers():
    q = "Summarize the preliminary experimental result comparing query strategies across both scenarios."
    assert classify_question(q).search_type == "similarity"


def test_router_choice_executes_exactly_one_retrieval():
    base = FakeDense()
    store = AdaptiveRetrievalStore(base)
    cross = "Both XPINN and SK-PINN address training cost."
    single = "What is the NTK?"
    assert len(store.as_retriever().invoke(cross)) == 1
    assert store.last_route["mode"] == "mmr"
    assert len(store.as_retriever().invoke(single)) == 1
    assert store.last_route["mode"] == "similarity"
    assert base.queries == [("mmr", cross), ("similarity", single)]
