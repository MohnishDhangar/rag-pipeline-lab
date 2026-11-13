import pytest
from rag_lab.routing.query_features import features, FEATURE_NAMES
from rag_lab.routing.adaptive_router import AdaptiveRouter
from rag_lab.routing.train_router import prepare_rows, train

class Strategy:
    def __init__(self,label): self.label=label
    def run(self,q): return self.label

def test_feature_shape():
    assert len(features("How do PINNs compare to graph networks?"))==len(FEATURE_NAMES)

def test_safe_router_fallback_without_training():
    router=AdaptiveRouter({"naive":Strategy("low cost"),"hyde":Strategy("more calls")})
    assert router.run("Who used this parameter?")=="low cost"
    assert router.last_route=="naive"

def test_training_requires_real_scored_examples(tmp_path):
    generations=[dict(strategy="naive",backend="dense",id="1",question="Q",answer="A")]
    with pytest.raises(ValueError,match="25"):
        train(generations,[dict(strategy="naive",backend="dense",id="1",
                                answer_correctness=.9)],tmp_path/"model.joblib")

def test_preparation_chooses_lower_call_count_on_tie():
    gen=[{"strategy":n,"backend":"dense","id":"1","question":"What is PINN?","answer":"A"}
         for n in ("naive","decomposition")]
    scores=[{"strategy":n,"backend":"dense","id":"1","answer_correctness":.8}
            for n in ("naive","decomposition")]
    assert prepare_rows(gen,scores)[0][2]=="naive"
