# 07 — Comprehensive Evaluation Strategy & LLM-as-Judge Calibration

## 1. Tri-Part Evaluation Framework

To establish whether the support agent is worthy of deployment trust, we evaluate the system along three rigorous, distinct axes:

```text
               EVALUATION SUITE
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
[Intent Metrics] [Escalation Metrics] [Reply Quality Metrics]
- Macro F1       - False Auto-Handle  - Groundedness (1-5)
- Accuracy         Rate (FAHR)        - Relevance (1-5)
- Per-Class PRF1 - Escalation Recall  - Hallucination Penalty
- Confusion      - Auto-Handle        - Human-Judge
  Matrix           Coverage             Calibration (Kappa / ρ)
```

---

## 2. Component A: Intent Classification Metrics

For classes $k \in \{1, \dots, K\}$:

1. **Accuracy**:
   $$\text{Accuracy} = \frac{\sum_{i=1}^N \mathbb{I}(y_i = \hat{y}_i)}{N}$$
   *Diagnostic Role*: Baseline overall correctness; susceptible to class imbalance distortion.

2. **Macro F1 (Primary Optimizing Metric)**:
   $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K F_{1, k} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$
   *Diagnostic Role*: Treats all intents equally, preventing the system from appearing high-performing while failing minority intents.

3. **Weighted F1**:
   $$\text{Weighted F1} = \sum_{k=1}^K w_k \cdot F_{1, k}, \quad w_k = \frac{N_k}{N}$$

4. **Confusion Matrix Analysis**:
   Generates a normalized $K \times K$ confusion matrix to identify systemic semantic confusion pairs (e.g., `account_access_auth` vs `billing_subscription`).

---

## 3. Component B: Escalation & Safety Metrics

In customer support, **under-escalation** (falsely auto-handling an issue that required human triage) is dangerous, while **over-escalation** (sending simple FAQs to humans) is merely inefficient.

Let:
* $N_{\text{auto}} = \sum \mathbb{I}(\text{Decision} = \text{AUTO\_HANDLE})$
* $N_{\text{esc}} = \sum \mathbb{I}(\text{Decision} = \text{ESCALATE})$
* $Y_{\text{esc}}^* \in \{0, 1\}$ be the gold ground-truth escalation need.

1. **False Auto-Handle Rate (FAHR — Critical Safety Metric)**:
   $$\text{FAHR} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Decision}_i = \text{AUTO\_HANDLE} \land Y_{\text{esc}, i}^* = 1)}{N_{\text{auto}}}$$
   *Target*: $< 5\%$ on test set; strictly $0\%$ on high-risk intents.

2. **Auto-Handle Coverage**:
   $$\text{Coverage} = \frac{N_{\text{auto}}}{N}$$
   *Target*: Measures business ROI. An agent with 0% FAHR but 0% coverage has zero utility.

3. **Escalation Precision & Recall**:
   $$\text{Precision}_{\text{esc}} = \frac{\sum \mathbb{I}(\text{Decision} = \text{ESC} \land Y^* = 1)}{N_{\text{esc}}}, \quad \text{Recall}_{\text{esc}} = \frac{\sum \mathbb{I}(\text{Decision} = \text{ESC} \land Y^* = 1)}{\sum Y^*}$$

4. **Unnecessary Escalation Rate (Over-Escalation)**:
   $$\text{UER} = \frac{\sum \mathbb{I}(\text{Decision} = \text{ESC} \land Y^* = 0)}{N_{\text{esc}}}$$

---

## 4. Component C: Reply Quality & LLM-as-Judge

### The 5-Dimension Evaluation Rubric (Scale 1 to 5)

| Dimension | Description | 1 (Failing) | 3 (Acceptable) | 5 (Exemplary) |
| :--- | :--- | :--- | :--- | :--- |
| **Groundedness** | Extent to which facts, links, and steps are derived from retrieved historical resolutions. | Hallucinates fake URLs, policies, or claims. | General steps align, but includes minor unsupported statements. | 100% of claims and steps are directly supported by evidence. |
| **Relevance** | Directness in addressing the customer's specific problem. | Irrelevant or answers a completely different issue. | Addresses the general area but misses specific details. | Directly and precisely targets the customer's exact symptom. |
| **Helpfulness & Actionability** | Clear, logical next steps provided to the customer. | Vague, passive, or dead-end. | Gives a generic link without context. | Clear, immediate troubleshooting step or verification question. |
| **Safety & Policy** | Absence of unauthorized promises (free replacements, guaranteed refunds). | Promises refund or unauthorized compensation. | Slightly ambiguous policy wording. | Completely policy-compliant and safe. |
| **Tone & Brand Voice** | Professional, empathetic, concise, and matching Twitter support norms. | Rude, robotic, overly lengthy, or aggressive. | Acceptable but overly formal/dry. | Empathetic, courteous, natural, concise ($\le 240$ chars). |

---

## 5. Validating the LLM Judge: Human-Judge Calibration

An uncalibrated LLM judge cannot be trusted to evaluate another model. We implement a **meta-evaluation protocol**:

1. **Calibration Subset**: A random subset of $N_{\text{cal}} = 50$ test instances is scored independently by a human annotator across all 5 dimensions.
2. **Judge Scoring**: The automated LLM judge scores the identical 50 instances using the standardized rubric prompt.
3. **Agreement Quantification**:
   * **Spearman Rank Correlation ($\rho$)**: Measures monotonic rank agreement across numerical ratings (1–5). Appropriate because ratings are ordinal:
     $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
   * **Cohen's Weighted Kappa ($\kappa_w$)**: Measures inter-rater agreement for categorical/ordinal bins, accounting for chance agreement:
     $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
   * **Absolute Mean Error**: Measures systematic judge leniency or harshness:
     $$\text{Judge Bias} = \frac{1}{N} \sum (\text{Score}_{\text{judge}} - \text{Score}_{\text{human}})$$
4. **Validation Acceptance Gate**: The LLM judge is accepted only if $\rho \ge 0.65$ and $\kappa_w \ge 0.60$ with humans.
