"""Unified Support Agent Pipeline orchestrating intent classification, retrieval, escalation, and grounded drafting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.data.cleaner import clean_tweet_text
from src.intents.embedding_classifier import EmbeddingIntentClassifier
from src.retrieval.index import ResolutionRetrievalIndex
from src.escalation.policy import EscalationPolicy, EscalationDecision
from src.generation.grounded_drafter import GroundedReplyDrafter


@dataclass
class AgentTriageResult:
    """Unified result emitted by the Support Agent Pipeline for an incoming query."""
    query: str
    cleaned_query: str
    predicted_intent: str
    intent_confidence: float
    intent_distribution: Dict[str, float]
    decision: str  # "AUTO_HANDLE" | "ESCALATE"
    reason_code: str
    reason_details: str
    draft_reply: str
    groundedness_score: float
    retrieval_similarity: float
    retrieved_evidence: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "cleaned_query": self.cleaned_query,
            "predicted_intent": self.predicted_intent,
            "intent_confidence": round(self.intent_confidence, 4),
            "intent_distribution": {k: round(v, 4) for k, v in self.intent_distribution.items()},
            "escalation": {
                "decision": self.decision,
                "reason_code": self.reason_code,
                "reason_details": self.reason_details,
            },
            "draft_reply": self.draft_reply,
            "groundedness_score": round(self.groundedness_score, 2),
            "retrieval_similarity": round(self.retrieval_similarity, 4),
            "retrieved_evidence": self.retrieved_evidence,
        }


class SupportAgentPipeline:
    """End-to-end AI Customer Support Agent pipeline."""

    def __init__(
        self,
        intent_classifier: Optional[EmbeddingIntentClassifier] = None,
        retrieval_index: Optional[ResolutionRetrievalIndex] = None,
        escalation_policy: Optional[EscalationPolicy] = None,
        drafter: Optional[GroundedReplyDrafter] = None,
        brand_handle: str = "AppleSupport"
    ):
        self.brand_handle = brand_handle
        self.classifier = intent_classifier or EmbeddingIntentClassifier()
        self.retrieval = retrieval_index or ResolutionRetrievalIndex()
        self.escalation = escalation_policy or EscalationPolicy()
        self.drafter = drafter or GroundedReplyDrafter(brand_handle=brand_handle)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Fits intent classifier, builds FAISS retrieval index from training pairs,

        and optionally tunes escalation thresholds on validation pairs.
        """
        print(f"[pipeline] Training intent classifier on {len(train_df)} training samples...")
        self.classifier.fit(train_df["customer_text"].tolist(), train_df["intent"].tolist())

        print(f"[pipeline] Building FAISS retrieval index from {len(train_df)} historical training pairs...")
        pairs_records = train_df.to_dict(orient="records")
        self.retrieval.build_index(pairs_records)

        if val_df is not None and len(val_df) > 0:
            print(f"[pipeline] Tuning escalation thresholds on {len(val_df)} validation samples...")
            val_records = []
            for _, r in val_df.iterrows():
                q = str(r["customer_text"])
                pred_intent, conf, _ = self.classifier.predict_single(q)
                hits = self.retrieval.search(q, top_k=1, intent_filter=pred_intent)
                top_sim = hits[0][1] if hits else 0.0
                val_records.append({
                    "query": q,
                    "intent": pred_intent,
                    "confidence": conf,
                    "retrieval_similarity": top_sim,
                    "gold_should_escalate": r.get("should_escalate", False)
                })
            best_ti, best_tr, best_fahr, best_cov = EscalationPolicy.tune_thresholds(val_records)
            self.escalation.tau_intent = best_ti
            self.escalation.tau_retrieval = best_tr
            print(f"[pipeline] Tuned thresholds: tau_intent={best_ti:.2f}, tau_retrieval={best_tr:.2f} (Val FAHR={best_fahr:.1%}, Coverage={best_cov:.1%})")

    def process(self, query: str) -> AgentTriageResult:
        """Processes an incoming customer query through the 6-stage pipeline."""
        clean_q = clean_tweet_text(query, preserve_brand_mention=self.brand_handle)

        # 1. Intent classification
        pred_intent, conf, dist = self.classifier.predict_single(clean_q)

        # 2. Historical resolution retrieval (intent-conditioned)
        hits = self.retrieval.search(clean_q, top_k=2, intent_filter=pred_intent)
        top_sim = hits[0][1] if hits else 0.0

        # 3. Grounded reply drafting
        draft_info = self.drafter.draft_reply(clean_q, pred_intent, hits)

        # 4. Escalation decision
        decision: EscalationDecision = self.escalation.evaluate(
            query=clean_q,
            predicted_intent=pred_intent,
            intent_confidence=conf,
            retrieval_similarity=top_sim,
            draft_reply=draft_info["draft_reply"]
        )

        evidence_list = []
        for doc, score in hits:
            evidence_list.append({
                "pair_id": doc.get("pair_id", ""),
                "historical_customer_query": doc.get("customer_text", ""),
                "historical_brand_reply": doc.get("brand_text", ""),
                "intent": doc.get("intent", ""),
                "similarity_score": round(score, 4),
            })

        return AgentTriageResult(
            query=query,
            cleaned_query=clean_q,
            predicted_intent=pred_intent,
            intent_confidence=conf,
            intent_distribution=dist,
            decision=decision.action,
            reason_code=decision.reason_code,
            reason_details=decision.details,
            draft_reply=draft_info["draft_reply"],
            groundedness_score=draft_info["groundedness_score"],
            retrieval_similarity=top_sim,
            retrieved_evidence=evidence_list
        )

    def save(self, model_dir: str = "results/models"):
        """Serializes fitted models and indices to disk."""
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.classifier.save(str(out / "intent_classifier.pkl"))
        self.retrieval.save(str(out / "retrieval_index"))

    @classmethod
    def load(cls, model_dir: str = "results/models", device: str = "cpu") -> "SupportAgentPipeline":
        """Loads fitted pipeline from disk."""
        inp = Path(model_dir)
        clf = EmbeddingIntentClassifier.load(str(inp / "intent_classifier.pkl"), device=device)
        ret = ResolutionRetrievalIndex.load(str(inp / "retrieval_index"), device=device)
        return cls(intent_classifier=clf, retrieval_index=ret)
