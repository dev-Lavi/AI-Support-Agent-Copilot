# 01 — Problem Framing & Scope

## 1. Executive Summary

Modern AI customer support systems frequently fail not from a lack of generative fluency, but from an inability to know **when they do not know**. In high-stakes customer service, hallucinating a non-existent refund policy, giving contradictory instructions, or falsely claiming an issue is resolved causes severe customer churn and operational risk.

This project builds a **trustworthy, retrieval-grounded, and conservatively escalating AI support agent** tailored to a single brand from the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`). The core thesis of this work is: **The proof is worth more than the system.** Every architectural decision prioritizes auditability, calibration, and safety over unconstrained automation.

---

## 2. Defining "Good" Across System Components

"Good" is not a monolithic number. An agent with 95% intent classification accuracy is dangerous if its 5% failure cases silently auto-resolve billing disputes with fabricated URLs. We decompose "good" into three orthogonal, measurable dimensions:

### A. Intent Classification: What Does "Good" Mean?

1. **Macro F1 > Accuracy**: In customer support datasets, query distributions follow a power law (e.g., general shipping questions dominate, while urgent account takeover issues are rare). A model predicting only the majority class achieves deceptively high accuracy while completely failing critical minority intents. "Good" means high **Macro F1** across all defined intents.
2. **Low Confusion on Critical Boundaries**: Confusing two benign informational classes (e.g., `store_hours` vs `location_query`) has low penalty. Confusing `cancellation_request` with `general_feedback` is catastrophic. "Good" implies a confusion matrix where high-consequence intents have near-zero false negatives.
3. **Robustness to Noisy Customer Input**: Real tweets are filled with typos, emojis, sarcasm, fragmented grammar, and all-caps frustration. A good classifier maintains calibrated probabilities despite input noise.
4. **Calibrated Posterior Probabilities**: The confidence score $P(\hat{y}|x)$ must reflect true empirical likelihood. If the model outputs 0.60 confidence, it should be correct ~60% of the time, allowing threshold-based escalation gates to function reliably.

### B. Reply Generation: What Does "Good" Mean?

1. **Retrieval Groundedness**: Every drafted reply must be factually supported by historical brand resolutions retrieved from the training corpus. The model must not invent new policies, contact channels, discount amounts, or turnaround times.
2. **Brand Voice & Style Alignment**: Customer support on Twitter has a distinct brand-specific register (e.g., empathetic tone, direct invitation to DM, explicit sign-offs like `^AB`). A good reply reproduces the brand's verified communication norms.
3. **Conciseness & Actionability**: Tweets have character constraints and users want immediate steps, not long-winded apologies. A good reply provides a clear next step (e.g., "Please DM us your 6-character booking reference").
4. **Zero Hallucination of Sensitive Entities**: The generator must never output fabricated phone numbers, broken links, unauthorized promises ("I have issued a full refund"), or fictitious employee names.
5. **Graceful Failure**: If the retrieved historical resolutions do not contain sufficient evidence to resolve the query, the generator must explicitly defer rather than attempt an ungrounded response.

### C. Escalation Decision: What Does "Good" Mean?

1. **Safety-First Conservative Bias**: It is far better to escalate an issue that a human could have handled easily (operational inefficiency) than to auto-handle an issue incorrectly (customer disaster).
2. **Minimizing False Auto-Handles (FAHR)**: The most critical safety metric is the **False Auto-Handle Rate**—the proportion of automated replies that required human intervention or were factually incorrect. In our framework:
   $$\text{FAHR} = \frac{\text{Auto-Handled Cases with Errors or Needing Human}}{\text{Total Auto-Handled Cases}}$$
   Our objective is to drive this metric toward zero.
3. **Explicit, Auditable Reason Codes**: The agent must never produce a silent binary escalation. Every escalation must be accompanied by a machine-parseable, human-understandable reason code (e.g., `LOW_INTENT_CONFIDENCE`, `INSUFFICIENT_HISTORICAL_EVIDENCE`, `SENSITIVE_ACCOUNT_ISSUE`).
4. **Viable Coverage**: An agent that escalates 100% of messages is 100% safe but delivers 0% automation value. "Good" means finding the optimal operating point on the **Coverage vs Risk** frontier on validation data.

---

## 3. What We Are NOT Building (Explicit Non-Goals)

To ensure depth and rigorous evaluation, we explicitly establish the project boundaries:

* **NOT a Full Human Replacement**: The system is designed as an automated first-line triage and draft copilot, not an autonomous agent operating without human oversight.
* **NOT a Multi-Brand Platform**: Support policies, customer tone, and vocabulary vary wildly between an airline, an e-commerce retailer, and a hardware manufacturer. We deliberately focus on **ONE selected brand** to achieve authentic grounding.
* **NOT an Open-Domain Conversational Chatbot**: We do not engage in casual chitchat, political discussions, or general knowledge Q&A. Out-of-scope inputs are routed to escalation or polite redirection.
* **NOT a Full Production Ticketing CRM**: We do not implement Kafka consumers, distributed microservices, or complex database connection pooling. We focus on ML/AI modeling, retrieval, inference, and rigorous evaluation.
* **NOT a Paid API Wrapper**: The entire core pipeline is engineered to run on free, open-source models (`sentence-transformers`, `scikit-learn`, `FAISS`, local HuggingFace transformers) reproducible on commodity hardware.

---

## 4. Epistemological Framework: Assumptions, Hypotheses & Measured Results

To ensure interview-grade intellectual honesty, all claims in this repository are categorized under one of four states:

| Category | Definition | Current Status in Project |
| :--- | :--- | :--- |
| **Assumption** | Foundational premises taken as given without formal testing. | The Twitter dataset reflects real customer interactions; training split historical replies represent approved brand behavior. |
| **Hypothesis** | Testable predictions formulated before running experiments. | Dense embeddings will outperform TF-IDF on noisy customer tweets; multi-signal escalation will reduce False Auto-Handles by >50% compared to single-signal confidence gating. |
| **Planned Approach** | The specific algorithmic or engineering design chosen for implementation. | Detailed in `docs/02-system-design.md` through `docs/08-baselines.md`. |
| **Measured Result** | Empirically verified quantitative outputs logged by the evaluation harness. | To be generated and populated strictly after pipeline execution on test and golden splits. |
