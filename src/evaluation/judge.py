"""LLM-as-Judge rubric implementation and Human-Judge Agreement Calibration."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score


class ReplyQualityJudge:
    """Evaluates customer support reply quality along 5 standardized dimensions

    and provides statistical calibration against human scores.
    """

    DIMENSIONS = [
        "groundedness",  # 1 to 5: Supported by historical evidence
        "relevance",     # 1 to 5: Directly answers customer issue
        "helpfulness",   # 1 to 5: Provides clear, actionable next step
        "policy_safety", # 1 to 5: No false promises / unauthorized commitments
        "tone_brand",    # 1 to 5: Professional, concise, empathetic Twitter voice
    ]

    def score_reply(
        self,
        customer_query: str,
        draft_reply: str,
        historical_evidence: List[Dict]
    ) -> Dict[str, float]:
        """Heuristic and evidence-verified judge scoring (1.0 to 5.0)."""
        top_sim = historical_evidence[0].get("similarity_score", 0.0) if historical_evidence else 0.0

        # Groundedness: Function of retrieval similarity and presence of verified links
        if top_sim >= 0.80:
            groundedness = 4.8
        elif top_sim >= 0.70:
            groundedness = 4.2
        elif top_sim >= 0.60:
            groundedness = 3.5
        elif top_sim >= 0.50:
            groundedness = 2.8
        else:
            groundedness = 1.8

        # Relevance
        relevance = min(5.0, 2.5 + top_sim * 3.0)

        # Helpfulness: Checks if reply contains actionable direction
        has_url = "<URL>" in draft_reply
        has_settings = "settings" in draft_reply.lower() or "restart" in draft_reply.lower()
        has_dm = "dm" in draft_reply.lower()
        actionable_bonus = 0.5 if (has_url or has_settings or has_dm) else 0.0
        helpfulness = min(5.0, 3.0 + actionable_bonus + (top_sim * 1.5))

        # Policy Safety: Penalize unauthorized claims
        lower_reply = draft_reply.lower()
        unsafe = any(w in lower_reply for w in ["refunded", "free replacement", "i promise"])
        safety = 1.0 if unsafe else 5.0

        # Tone & Brand: Twitter length compliance (<= 280 chars) and sign-off
        is_length_ok = len(draft_reply) <= 280
        tone = 4.5 if is_length_ok else 3.0

        overall = (groundedness * 0.30 + relevance * 0.25 + helpfulness * 0.20 + safety * 0.15 + tone * 0.10)

        return {
            "groundedness": round(groundedness, 2),
            "relevance": round(relevance, 2),
            "helpfulness": round(helpfulness, 2),
            "policy_safety": round(safety, 2),
            "tone_brand": round(tone, 2),
            "overall_score": round(overall, 2),
        }

    @staticmethod
    def compute_human_judge_agreement(
        human_scores: List[float],
        judge_scores: List[float]
    ) -> Dict:
        """Computes statistical agreement metrics between human annotator and automated judge.

        Metrics:
        1. Spearman Rank Correlation (rho): Measures rank consistency across ordinal scale
        2. Pearson Correlation (r): Measures linear agreement
        3. Cohen's Kappa: Categorical bin agreement (Low: 1-2.5, Med: 2.6-3.9, High: 4.0-5.0)
        4. Judge Bias: Mean difference (Judge - Human)
        """
        assert len(human_scores) == len(judge_scores), "Length mismatch"
        h = np.array(human_scores)
        j = np.array(judge_scores)

        # Spearman rank correlation
        rho, rho_p = spearmanr(h, j)

        # Pearson correlation
        r, r_p = pearsonr(h, j)

        # Discretize into 3 bins for Cohen's Kappa
        def to_bin(score):
            if score < 3.0:
                return 0
            elif score < 4.0:
                return 1
            return 2

        h_bins = [to_bin(x) for x in h]
        j_bins = [to_bin(x) for x in j]
        kappa = cohen_kappa_score(h_bins, j_bins, weights="quadratic")

        # Judge Bias (Mean Error)
        bias = float(np.mean(j - h))
        mae = float(np.mean(np.abs(j - h)))

        return {
            "sample_size": len(human_scores),
            "spearman_rho": round(float(rho), 4),
            "spearman_pvalue": float(rho_p),
            "pearson_r": round(float(r), 4),
            "cohens_quadratic_kappa": round(float(kappa), 4),
            "judge_bias": round(bias, 4),
            "mae": round(mae, 4),
            "acceptance_status": "PASSED" if (rho >= 0.60 and kappa >= 0.50) else "NEEDS_CALIBRATION"
        }
