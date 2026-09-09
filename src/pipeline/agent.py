"""Unified Support Agent Pipeline orchestrating intent classification, retrieval, escalation, and grounded drafting.

Supports high-accuracy Groq LLaMA-3.3-70B inference, zero-PyTorch Scikit-Learn TF-IDF fallback (<50MB RAM),
and optional dense PyTorch embeddings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.data.cleaner import clean_tweet_text
from src.escalation.policy import EscalationPolicy, EscalationDecision
from src.generation.grounded_drafter import GroundedReplyDrafter
from src.retrieval.tfidf_retriever import TfidfRetrievalIndex
from src.intents.tfidf_baseline import TfidfBaselineClassifier
from src.pipeline.groq_agent import GroqSupportAgent


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
        intent_classifier: Optional[Any] = None,
        retrieval_index: Optional[Any] = None,
        escalation_policy: Optional[EscalationPolicy] = None,
        drafter: Optional[GroundedReplyDrafter] = None,
        groq_agent: Optional[GroqSupportAgent] = None,
        brand_handle: str = "AppleSupport"
    ):
        self.brand_handle = brand_handle
        self.classifier = intent_classifier or TfidfBaselineClassifier()
        self.retrieval = retrieval_index or TfidfRetrievalIndex()
        self.escalation = escalation_policy or EscalationPolicy()
        self.drafter = drafter or GroundedReplyDrafter(brand_handle=brand_handle)
        self.groq_agent = groq_agent or GroqSupportAgent()

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Fits intent classifier, builds retrieval index from training pairs,

        and optionally tunes escalation thresholds on validation pairs.
        """
        print(f"[pipeline] Training intent classifier on {len(train_df)} training samples...")
        self.classifier.fit(train_df["customer_text"].tolist(), train_df["intent"].tolist())

        print(f"[pipeline] Building retrieval index from {len(train_df)} historical training pairs...")
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

        # 1. Historical resolution retrieval
        hits = self.retrieval.search(clean_q, top_k=2)
        top_sim = hits[0][1] if hits else 0.0

        evidence_list = []
        for doc, score in hits:
            evidence_list.append({
                "pair_id": doc.get("pair_id", ""),
                "historical_customer_query": doc.get("customer_text", ""),
                "historical_brand_reply": doc.get("brand_text", ""),
                "intent": doc.get("intent", ""),
                "similarity_score": round(score, 4),
            })

        # 2. If Groq LLM is configured, invoke Groq with retrieved evidence for high accuracy
        if self.groq_agent and self.groq_agent.is_configured:
            try:
                g_res = self.groq_agent.triage(clean_q, retrieved_evidence=hits)
                return AgentTriageResult(
                    query=query,
                    cleaned_query=clean_q,
                    predicted_intent=g_res["predicted_intent"],
                    intent_confidence=g_res["intent_confidence"],
                    intent_distribution=g_res["intent_distribution"],
                    decision=g_res["decision"],
                    reason_code=g_res["reason_code"],
                    reason_details=g_res["reason_details"],
                    draft_reply=g_res["draft_reply"],
                    groundedness_score=g_res["groundedness_score"],
                    retrieval_similarity=top_sim,
                    retrieved_evidence=evidence_list
                )
            except Exception as e:
                print(f"[pipeline] Groq API warning: {e}. Falling back to local offline engine...")

        # 3. Local Fallback: Intent classification
        pred_intent, conf, dist = self.classifier.predict_single(clean_q)

        # Filtered retrieval if intent is known
        filtered_hits = self.retrieval.search(clean_q, top_k=2, intent_filter=pred_intent)
        if filtered_hits:
            hits = filtered_hits
            top_sim = hits[0][1]

        # 4. Grounded reply drafting
        draft_info = self.drafter.draft_reply(clean_q, pred_intent, hits)

        # 5. Escalation decision
        decision: EscalationDecision = self.escalation.evaluate(
            query=clean_q,
            predicted_intent=pred_intent,
            intent_confidence=conf,
            retrieval_similarity=top_sim,
            draft_reply=draft_info["draft_reply"]
        )

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
        """Serializes models and indices to disk."""
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        if hasattr(self.classifier, "save"):
            self.classifier.save(str(out / "tfidf_classifier.pkl"))
        if hasattr(self.retrieval, "save"):
            self.retrieval.save(str(out / "retrieval_index"))

    @classmethod
    def load(
        cls,
        model_dir: str = "results/models",
        prefer_lightweight: bool = True,
        device: str = "cpu"
    ) -> "SupportAgentPipeline":
        """Loads pipeline from disk. Uses ultra-lightweight TF-IDF (<40MB RAM) by default."""
        inp = Path(model_dir)

        # Check if user explicitly wants heavy PyTorch model via environment
        use_torch = os.environ.get("USE_TORCH", "").lower() in ["1", "true", "yes"]

        if not use_torch or prefer_lightweight:
            ret_index = TfidfRetrievalIndex.load(str(inp / "retrieval_index"))
            tfidf_path = inp / "tfidf_classifier.pkl"
            if tfidf_path.exists():
                clf = TfidfBaselineClassifier.load(str(tfidf_path))
            else:
                meta_file = inp / "retrieval_index" / "retrieval_metadata.json"
                import json
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                clf = TfidfBaselineClassifier().fit(
                    [d["customer_text"] for d in meta],
                    [d["intent"] for d in meta]
                )
            groq = GroqSupportAgent()
            return cls(intent_classifier=clf, retrieval_index=ret_index, groq_agent=groq)

        # Fallback to local PyTorch if explicitly requested
        from src.intents.embedding_classifier import EmbeddingIntentClassifier
        from src.retrieval.index import ResolutionRetrievalIndex
        clf = EmbeddingIntentClassifier.load(str(inp / "intent_classifier.pkl"), device=device)
        ret = ResolutionRetrievalIndex.load(str(inp / "retrieval_index"), device=device)
        ret._encoder = clf.encoder
        return cls(intent_classifier=clf, retrieval_index=ret)
