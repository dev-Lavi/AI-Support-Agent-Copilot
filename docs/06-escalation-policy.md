# 06 — Escalation Policy & Calibration

## 1. Safety-Centric Decision Philosophy

In customer support automation, the primary failure mode is **overconfidence on ambiguous or sensitive queries**. A support agent that attempts to answer every inquiry inevitably causes customer outrage on edge cases.

Our escalation policy enforces a **multi-signal conservative gating mechanism**. Automated handling is permitted if and only if the system demonstrates high statistical confidence across all decision layers:

$$\text{Decision}(x) = \begin{cases} \text{AUTO\_HANDLE}, & \text{if all safety criteria are met} \\ \text{ESCALATE}, & \text{if any safety criterion fails} \end{cases}$$

---

## 2. The Multi-Signal Escalation Function

For an incoming message $x$, let:
* $\hat{y} = \text{Predicted Intent}$
* $c_{\text{intent}} = P(\hat{y}|x) \in [0, 1]$ (Classifier Posterior Confidence)
* $S_{\text{retrieval}} = \max_i \text{CosineSimilarity}(e(x), e(q_i^{\text{hist}})) \in [-1, 1]$
* $\mathcal{S}_{\text{sensitive}} = \{ \text{account\_access\_auth}, \text{feedback\_complaint}, \text{other\_unknown} \}$
* $\mathcal{K}_{\text{urgent}} = \{ \text{"fraud"}, \text{"stolen"}, \text{"lawsuit"}, \text{"attorney"}, \text{"unauthorized charge"}, \text{"hack"} \}$

The formal decision chain evaluates in strict priority order:

```python
def decide_escalation(x, intent, c_intent, s_retrieval, draft_reply, tau_intent, tau_retrieval):
    # 1. Unknown or Catch-All Intent Check
    if intent == "other_unknown":
        return EscalationResult(
            action="ESCALATE", 
            reason="UNKNOWN_INTENT",
            details="Message cannot be reliably mapped to an operational intent."
        )

    # 2. Sensitive / Policy-Restricted Intents
    if intent in SENSITIVE_INTENTS:
        return EscalationResult(
            action="ESCALATE",
            reason="SENSITIVE_CASE",
            details=f"Intent '{intent}' involves security, legal, or high-touch human handling."
        )

    # 3. Urgent / High-Risk Keyword Trigger
    if any(keyword in x.lower() for keyword in URGENT_KEYWORDS):
        return EscalationResult(
            action="ESCALATE",
            reason="HIGH_RISK_KEYWORD_DETECTED",
            details="Customer query contains emergency, legal, or fraud indicators."
        )

    # 4. Intent Classification Confidence Threshold
    if c_intent < tau_intent:
        return EscalationResult(
            action="ESCALATE",
            reason="LOW_INTENT_CONFIDENCE",
            details=f"Classifier confidence {c_intent:.3f} below operating threshold {tau_intent:.2f}."
        )

    # 5. Historical Retrieval Evidence Similarity Threshold
    if s_retrieval < tau_retrieval:
        return EscalationResult(
            action="ESCALATE",
            reason="INSUFFICIENT_HISTORICAL_EVIDENCE",
            details=f"Top retrieval similarity {s_retrieval:.3f} below evidence threshold {tau_retrieval:.2f}."
        )

    # 6. Post-Generation Hallucination / Guardrail Trigger
    if not passes_guardrails(draft_reply):
        return EscalationResult(
            action="ESCALATE",
            reason="GENERATION_GUARDRAIL_FAILED",
            details="Draft reply contains unverified entities or policy violations."
        )

    # All criteria passed: Safe to auto-handle
    return EscalationResult(
        action="AUTO_HANDLE",
        reason="HIGH_CONFIDENCE_GROUNDED",
        details="High classifier confidence and strong grounded historical precedent."
    )
```

---

## 3. Human-Readable Reason Codes

| Reason Code | Trigger Condition | Intended Human Routing |
| :--- | :--- | :--- |
| `UNKNOWN_INTENT` | Intent predicted as `other_unknown`. | Tier 1 General Ingestion Queue |
| `SENSITIVE_CASE` | Query belongs to `account_access_auth` or `feedback_complaint`. | Account Security / Brand Escalations |
| `HIGH_RISK_KEYWORD` | Urgent keywords (fraud, legal, stolen, hack). | Senior Supervisory Queue |
| `LOW_INTENT_CONFIDENCE` | $P(\hat{y}\|x) < \tau_{\text{intent}}$ (ambiguous phrasing). | Intent Annotation / Tier 1 Support |
| `INSUFFICIENT_HISTORICAL_EVIDENCE` | $S_{\text{retrieval}} < \tau_{\text{retrieval}}$ (novel issue). | Knowledge Base Authoring Team |
| `CONFLICTING_EVIDENCE` | Top retrieval candidates propose opposite resolution actions. | Technical Specialist Queue |
| `GENERATION_GUARDRAIL_FAILED` | Hallucinated link, ungrounded entity, or tone issue. | Quality Assurance / System Audit |

---

## 4. Parameter Tuning Methodology on Validation Data

Thresholds $\tau_{\text{intent}}$ and $\tau_{\text{retrieval}}$ are **not hardcoded guesses**. They are empirically tuned on the Validation split ($\mathcal{D}_{\text{val}}$) by optimizing a multi-objective cost function:

$$\min_{\tau_{\text{intent}}, \tau_{\text{retrieval}}} \left[ \lambda \cdot \text{FalseAutoHandleRate}(\tau) + (1 - \lambda) \cdot (1 - \text{Coverage}(\tau)) \right]$$

where:
* **Coverage**: Proportion of queries auto-handled:
  $$\text{Coverage} = \frac{\sum \mathbb{I}(\text{Decision} = \text{AUTO\_HANDLE})}{N}$$
* **False Auto-Handle Rate (FAHR)**: The critical safety metric:
  $$\text{FAHR} = \frac{\sum \mathbb{I}(\text{Decision} = \text{AUTO\_HANDLE} \land \text{Gold} = \text{ESCALATE})}{\sum \mathbb{I}(\text{Decision} = \text{AUTO\_HANDLE})}$$
* $\lambda \in [0, 1]$ is the safety penalty weight ($\lambda = 0.85$), heavily penalizing false auto-handles over lower coverage.
