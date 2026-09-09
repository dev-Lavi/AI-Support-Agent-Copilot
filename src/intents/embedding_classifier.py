"""Final Intent Classifier: Dense Semantic Embeddings + Calibrated Classification Head."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.intents.taxonomy import INTENTS, INTENT2ID, ID2INTENT


class EmbeddingIntentClassifier:
    """Dense neural intent classifier utilizing all-MiniLM-L6-v2 sentence embeddings

    with a softmax-calibrated classification head.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._encoder = None
        self.clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        self.is_fitted: bool = False

    @property
    def encoder(self):
        """Lazy loader for SentenceTransformer to minimize startup memory."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
        return self._encoder

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Computes normalized dense embeddings for texts."""
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings

    def fit(self, texts: List[str], labels: List[str]) -> "EmbeddingIntentClassifier":
        """Fits the classifier head on dense text embeddings."""
        X = self.encode(texts)
        y = np.array([INTENT2ID[lbl] for lbl in labels])
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts the intent class label for input texts."""
        assert self.is_fitted, "Model must be fitted before predict"
        X = self.encode(texts)
        preds = self.clf.predict(X)
        return [ID2INTENT[p] for p in preds]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns calibrated posterior class probabilities."""
        assert self.is_fitted, "Model must be fitted before predict_proba"
        X = self.encode(texts)
        raw_probs = self.clf.predict_proba(X)

        n = len(texts)
        full_probs = np.zeros((n, len(INTENTS)), dtype=np.float32)
        for local_col, class_idx in enumerate(self.clf.classes_):
            full_probs[:, class_idx] = raw_probs[:, local_col]
        return full_probs

    def predict_single(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Convenience method for single-query inference."""
        probs = self.predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_intent = ID2INTENT[pred_idx]
        intent_distribution = {INTENTS[i]: float(probs[i]) for i in range(len(INTENTS))}
        return pred_intent, confidence, intent_distribution

    def save(self, filepath: str):
        """Saves fitted classifier parameters to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model_name": self.model_name,
                "clf": self.clf
            }, f)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "EmbeddingIntentClassifier":
        """Loads fitted classifier parameters from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls(model_name=data["model_name"], device=device)
        obj.clf = data["clf"]
        obj.is_fitted = True
        return obj
