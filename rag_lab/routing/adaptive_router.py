"""Cost-aware query router: zero GPT calls to choose a strategy."""
from pathlib import Path
from .query_features import features, FEATURE_NAMES

class AdaptiveRouter:
    def __init__(self, strategies, model_path=None, default="naive", threshold=0.40):
        if not strategies: raise ValueError("At least one strategy required")
        self.strategies = strategies
        self.default = default if default in strategies else next(iter(strategies))
        self.threshold=threshold
        self.last_route=None
        self.estimator=None
        if model_path:
            import joblib
            # Only load trusted locally generated joblib artifacts, never user uploads.
            artifact=joblib.load(Path(model_path))
            if artifact["features"] != FEATURE_NAMES:
                raise ValueError("Router feature contract mismatch")
            self.estimator=artifact["model"]

    def choose(self, question):
        selected=self.default
        if self.estimator:
            x=[features(question)]
            prediction=self.estimator.predict(x)[0]
            if prediction in self.strategies:
                probabilities=self.estimator.predict_proba(x)[0]
                labels=self.estimator.classes_
                confidence=float(probabilities[list(labels).index(prediction)])
                if confidence>=self.threshold:
                    selected=prediction
        self.last_route=selected
        return selected

    def run(self, question):
        return self.strategies[self.choose(question)].run(question)

    def retrieve(self, question):
        return self.strategies[self.choose(question)].retrieve(question)

    def run_with_trace(self, question):
        chosen=self.choose(question)
        result=self.strategies[chosen].run_with_trace(question)
        result.trace["selected_route"]=chosen
        result.trace["router"]="learned" if self.estimator else "conservative-fallback"
        return result
