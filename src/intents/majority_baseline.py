"""Baseline 1: Trivial Majority-Class Classifier."""

from typing import List, Dict, Optional
import numpy as np
from sklearn.dummy import DummyClassifier

from src.intents.taxonomy import INTENTS, INTENT2ID, ID2INTENT


class MajorityClassBaseline:
    """Trivial baseline that unconditionally predicts the most frequent class in the training set."""

    def __init__(self):
        self.clf = DummyClassifier(strategy="most_frequent")
        self.majority_intent_: Optional[str] = None
        self.majority_idx_: Optional[int] = None
        self.is_fitted: bool = False

    def fit(self, texts: List[str], labels: List[str]) -> "MajorityClassBaseline":
        """Fits the majority classifier on training labels."""
        y_ids = np.array([INTENT2ID[lbl] for lbl in labels])
        self.clf.fit(texts, y_ids)
        self.majority_idx_ = int(self.clf.predict([texts[0]])[0])
        self.majority_intent_ = ID2INTENT[self.majority_idx_]
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts the majority intent for all input texts."""
        assert self.is_fitted, "Model must be fitted before predict"
        preds = self.clf.predict(texts)
        return [ID2INTENT[p] for p in preds]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns uniform or one-hot probabilities."""
        assert self.is_fitted, "Model must be fitted before predict_proba"
        n = len(texts)
        probs = np.zeros((n, len(INTENTS)), dtype=np.float32)
        probs[:, self.majority_idx_] = 1.0
        return probs
