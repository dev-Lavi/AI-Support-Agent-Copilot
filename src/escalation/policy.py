"""Multi-Signal Conservative Escalation Decision Policy and Threshold Calibration."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from src.intents.taxonomy import SENSITIVE_INTENTS


# High-risk keywords indicating legal action, physical safety hazard, or financial fraud
URGENT_KEYWORDS = [
    "lawsuit", "lawyer", "attorney", "sue", "scam", "fraud", "stolen", "stole",
    "swelling", "melted", "burned", "smoke", "sparks", "fire",
    "unauthorized", "unauthorized charge", "refund", "dispute", "hacked", "police", "court"
]


@dataclass
class EscalationDecision:
    """Represents the structured output of the escalation engine."""
    action: str  # "AUTO_HANDLE" | "ESCALATE"
    reason_code: str  # Machine-readable reason code
    details: str  # Human-understandable rationale
    intent_confidence: float
    retrieval_similarity: float
    is_sensitive_intent: bool
    risk_keyword_matched: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "reason_code": self.reason_code,
            "details": self.details,
            "intent_confidence": round(self.intent_confidence, 4),
            "retrieval_similarity": round(self.retrieval_similarity, 4),
            "is_sensitive_intent": self.is_sensitive_intent,
            "risk_keyword_matched": self.risk_keyword_matched,
        }


class EscalationPolicy:
    """Evaluates multi-signal conservative escalation rules."""

    def __init__(
        self,
        tau_intent: float = 0.28,
        tau_retrieval: float = 0.20,
        sensitive_intents: Optional[List[str]] = None
    ):
        self.tau_intent = tau_intent
        self.tau_retrieval = tau_retrieval
        self.sensitive_intents = set(sensitive_intents or SENSITIVE_INTENTS)

    def evaluate(
        self,
        query: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_similarity: float,
        draft_reply: Optional[str] = None
    ) -> EscalationDecision:
        """Evaluates the input signals against the multi-stage conservative escalation ladder.

        Priority order:
        1. Unknown intent
        2. Sensitive intent category
        3. High-risk keywords
        4. Low intent classifier confidence
        5. Insufficient historical retrieval evidence
        6. Guardrail validation
        7. Auto-handle
        """
        query_lower = query.lower()

        # 1. Unknown / Catch-all intent check
        if predicted_intent == "other_unknown":
            return EscalationDecision(
                action="ESCALATE",
                reason_code="UNKNOWN_INTENT",
                details="Query could not be mapped to an operational support category.",
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                is_sensitive_intent=True
            )

        # 2. Sensitive / Policy-restricted intent check
        if predicted_intent in self.sensitive_intents:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="SENSITIVE_CASE",
                details=f"Intent '{predicted_intent}' involves security, legal, or high-touch human handling.",
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                is_sensitive_intent=True
            )

        # 3. Urgent / High-Risk keyword scan
        for kw in URGENT_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                return EscalationDecision(
                    action="ESCALATE",
                    reason_code="HIGH_RISK_KEYWORD_DETECTED",
                    details=f"Detected urgent safety/legal keyword: '{kw}'.",
                    intent_confidence=intent_confidence,
                    retrieval_similarity=retrieval_similarity,
                    is_sensitive_intent=True,
                    risk_keyword_matched=kw
                )

        # 4. Intent classification confidence threshold
        if intent_confidence < self.tau_intent:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="LOW_INTENT_CONFIDENCE",
                details=f"Intent confidence ({intent_confidence:.3f}) is below operating threshold ({self.tau_intent:.2f}).",
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                is_sensitive_intent=False
            )

        # 5. Historical retrieval evidence similarity threshold
        if retrieval_similarity < self.tau_retrieval:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="INSUFFICIENT_HISTORICAL_EVIDENCE",
                details=f"Nearest historical precedent similarity ({retrieval_similarity:.3f}) is below evidence threshold ({self.tau_retrieval:.2f}).",
                intent_confidence=intent_confidence,
                retrieval_similarity=retrieval_similarity,
                is_sensitive_intent=False
            )

        # 6. Post-generation guardrail check (if reply draft is provided)
        if draft_reply:
            # Check for unauthorized financial refund promises
            unauth_patterns = [r"\bi have refunded\b", r"\bfull refund issued\b", r"\bwe will pay\b"]
            for pat in unauth_patterns:
                if re.search(pat, draft_reply.lower()):
                    return EscalationDecision(
                        action="ESCALATE",
                        reason_code="GENERATION_GUARDRAIL_FAILED",
                        details="Generated draft contained an unauthorized commitment.",
                        intent_confidence=intent_confidence,
                        retrieval_similarity=retrieval_similarity,
                        is_sensitive_intent=False
                    )

        # All criteria satisfied: Safe to auto-handle
        return EscalationDecision(
            action="AUTO_HANDLE",
            reason_code="HIGH_CONFIDENCE_GROUNDED",
            details="High classification confidence and verified historical precedent.",
            intent_confidence=intent_confidence,
            retrieval_similarity=retrieval_similarity,
            is_sensitive_intent=False
        )

    @staticmethod
    def tune_thresholds(
        val_records: List[Dict],
        safety_weight: float = 0.85
    ) -> Tuple[float, float, float, float]:
        """Empirically tunes (tau_intent, tau_retrieval) on validation records to minimize

        False Auto-Handle Rate (FAHR) while preserving viable coverage.

        Returns:
            (best_tau_intent, best_tau_retrieval, best_fahr, best_coverage)
        """
        candidate_tau_intent = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        candidate_tau_retrieval = [0.50, 0.55, 0.60, 0.65, 0.70]

        best_cost = float("inf")
        best_params = (0.70, 0.65, 0.0, 0.0)

        for ti in candidate_tau_intent:
            for tr in candidate_tau_retrieval:
                policy = EscalationPolicy(tau_intent=ti, tau_retrieval=tr)
                n_auto = 0
                n_false_auto = 0
                n_total = len(val_records)

                for r in val_records:
                    dec = policy.evaluate(
                        query=r["query"],
                        predicted_intent=r["intent"],
                        intent_confidence=r["confidence"],
                        retrieval_similarity=r["retrieval_similarity"]
                    )
                    gold_should_esc = r.get("gold_should_escalate", False)

                    if dec.action == "AUTO_HANDLE":
                        n_auto += 1
                        if gold_should_esc:
                            n_false_auto += 1

                coverage = n_auto / n_total if n_total > 0 else 0.0
                fahr = (n_false_auto / n_auto) if n_auto > 0 else 0.0

                # Multi-objective cost: penalty on FAHR + penalty on lost coverage
                cost = safety_weight * fahr + (1.0 - safety_weight) * (1.0 - coverage)
                if cost < best_cost:
                    best_cost = cost
                    best_params = (ti, tr, fahr, coverage)

        return best_params
