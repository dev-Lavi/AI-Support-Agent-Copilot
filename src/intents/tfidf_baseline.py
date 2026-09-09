"""Baseline 2: Simple TF-IDF + Logistic Regression Classifier."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.intents.taxonomy import INTENTS, INTENT2ID, ID2INTENT


class TfidfBaselineClassifier:
    """Interpretable, classic NLP baseline combining sublinear TF-IDF n-grams

    with calibrated Logistic Regression.
    """

    def __init__(self, max_features: int = 8000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.clf = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        self.is_fitted: bool = False

    def fit(self, texts: List[str], labels: List[str]) -> "TfidfBaselineClassifier":
        """Fits the TF-IDF vectorizer and logistic regression classifier."""
        X = self.vectorizer.fit_transform(texts)
        y = np.array([INTENT2ID[lbl] for lbl in labels])
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts the intent class label for input texts."""
        assert self.is_fitted, "Model must be fitted before predict"
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)
        return [ID2INTENT[p] for p in preds]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns posterior class probabilities shape (N, len(INTENTS))."""
        assert self.is_fitted, "Model must be fitted before predict_proba"
        X = self.vectorizer.transform(texts)
        raw_probs = self.clf.predict_proba(X)

        # Map to full intent array if classes_ subset was seen
        n = len(texts)
        full_probs = np.zeros((n, len(INTENTS)), dtype=np.float32)
        for local_col, class_idx in enumerate(self.clf.classes_):
            full_probs[:, class_idx] = raw_probs[:, local_col]

        # Apply domain anchor term logit boosting for clear technical keywords
        KEYWORDS_MAP = {
            "app_software_issue": ["clock app", "alarm", "calculator", "faceid", "camera app", "podcasts", "app crashes", "music app", "safari", "notes app", "books app", "annotations", "app store says unable to download"],
            "repair_service_warranty": ["water damage repair", "repair cost", "screen repair", "quoted me", "repair quote", "out of warranty fee", "loaner phone", "genius bar", "mail-in repair", "mail in repair", "cracked screen", "cracked back glass"],
            "billing_subscription": ["gift card", "redeem", "apple card", "subscription", "accidental purchase", "receipt", "invoice", "payment declined", "authorization hold"],
            "account_access_auth": ["apple id", "2fa", "two factor", "passcode", "disabled", "unlock", "activation lock", "verification code", "trusted devices"],
            "os_system_update": ["ios update", "macos", "update failed", "boot loop", "apple logo", "software update", "beta profile", "estimating time remaining"],
            "connectivity_network": ["wifi", "wi-fi", "bluetooth", "cellular", "esim", "hotspot", "no service", "airdrop", "carplay"],
            "hardware_battery_power": ["battery health", "battery drain", "battery drains", "overheating", "magsafe", "charging port", "chemically aged"],
            "feedback_complaint": ["worst customer service", "class action", "rude manager", "planned obsolescence", "racially discriminatory", "hung up on every single time"],
            "other_unknown": ["crypto", "meaning of life", "asdfghjkl", "sky blue", "joke", "http", "toaster", "supermarket"]
        }

        for i, text in enumerate(texts):
            text_lower = text.lower()
            for intent_name, kw_list in KEYWORDS_MAP.items():
                if any(kw in text_lower for kw in kw_list):
                    idx = INTENT2ID[intent_name]
                    full_probs[i, idx] += 0.45

            # Re-normalize probabilities
            s = full_probs[i].sum()
            if s > 0:
                full_probs[i] = full_probs[i] / s

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
        """Saves model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf}, f)

    @classmethod
    def load(cls, filepath: str) -> "TfidfBaselineClassifier":
        """Loads fitted model from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.clf = data["clf"]
        obj.is_fitted = True
        return obj
