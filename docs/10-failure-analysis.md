# 10 — Failure Analysis & Error Diagnostics

## 1. Diagnostic Framework

A machine learning system cannot be trusted until its failure boundaries are thoroughly understood. Rather than presenting generic aggregated errors, this document establishes a structured protocol for auditing the **Top 5 failure modes** discovered during evaluation.

Each failure is analyzed across eight mandatory dimensions:
1. **Failure Mode Name**
2. **Real Customer Input Example**
3. **Expected Behavior**
4. **Actual System Output**
5. **Root Cause Analysis (Why it happened)**
6. **Empirical Evidence**
7. **Underlying Hypothesis**
8. **Concrete Remediation / Fix**

---

## 2. Top 5 Measured Failure Modes (Empirical Evidence from Golden Set)

The following 5 failure cases represent actual empirical errors logged during evaluation of the 200-example Golden Set:

---

### Failure 1: Overconfidence on Ultra-Short Ambiguous Queries
* **Customer Input**: `"Battery drain"`
* **Predicted Intent**: `hardware_battery_power` (Confidence: 89.7%, Retrieval Sim: 0.691)
* **Actual System Decision**: `AUTO_HANDLE`
* **Expected Behavior**: `ESCALATE` (Reason: `AMBIGUOUS_QUERY` / Request Clarification)
* **Actual Output**: *"We want your battery to perform at its best. Check your Battery Health under Settings > Battery: <URL> and let us know what you see! ^AB"*
* **Why It Happened**: Two-word queries lack crucial context (device model, iOS version, sudden vs gradual drain). The classifier recognized the dominant keyword "battery" with high confidence, and semantic retrieval found a generic battery health precedent, passing the auto-handle threshold.
* **Hypothesis**: High posterior confidence on terse inputs is an artifact of closed-set classification without an input entropy / length regularizer.
* **Potential Fix**: Add an input length check in the escalation engine: if word count $< 4$ and query lacks a specific symptom, flag `AMBIGUOUS_QUERY` and escalate or ask a clarifying question.

---

### Failure 2: High-Stakes Data Loss Masked Under General App Software Intent
* **Customer Input**: `"Notes app wiped all my folders after the last update!! 5 years of work gone HELP ME PLEASE"`
* **Predicted Intent**: `app_software_issue` (Confidence: 83.8%, Retrieval Sim: 0.793)
* **Actual System Decision**: `AUTO_HANDLE`
* **Expected Behavior**: `ESCALATE` (Reason: `SENSITIVE_DATA_LOSS` / Supervisory Escalation)
* **Actual Output**: *"To fix the camera display, try a force restart using this guide: <URL>. Does the flashlight return after rebooting? ^CD"*
* **Why It Happened**: `app_software_issue` is classified as auto-handle eligible. While the intent was technically correct (Notes app), the severity of the issue (5 years of deleted work + high customer distress) required human intervention. Vector retrieval suffered lexical distraction, fetching a camera force-restart precedent.
* **Hypothesis**: Semantic embeddings alone do not encode emotional valence or business impact severity.
* **Potential Fix**: Introduce a distress/urgency keyword filter (`"lost work"`, `"wiped"`, `"deleted all"`, `"help me please"`) that triggers mandatory escalation regardless of intent confidence.

---

### Failure 3: Bricked Device / Critical Boot Loop Misclassified as Transient App Crash
* **Customer Input**: `"Phone is completely frozen on black screen with spinning gear wheel for 6 hours"`
* **Predicted Intent**: `app_software_issue` (Confidence: 80.7%, Retrieval Sim: 0.816)
* **Actual System Decision**: `AUTO_HANDLE`
* **Expected Behavior**: `ESCALATE` (Reason: `CRITICAL_SYSTEM_FREEZE` / Recovery Mode Triage)
* **Actual Output**: *"Let's get Netflix working again. Try force quitting the app, checking for updates in App Store, and restarting: <URL> ^AB"*
* **Why It Happened**: A black screen with spinning gear for 6 hours is a major OS boot failure (SpringBoard crash), not a third-party app issue. The retrieval index matched an app freeze precedent ("force quit and restart") which is completely ineffective for a device that won't boot.
* **Hypothesis**: The word "frozen" had high affinity to app freezing in the training corpus, overriding the hardware symptom "spinning gear".
* **Potential Fix**: Pre-filter queries containing `"black screen"`, `"spinning wheel"`, `"boot loop"`, or `"bricked"` into `os_system_update` with mandatory Recovery Mode instructions or human escalation.

---

### Failure 4: Financial Refund Disputes Auto-Handled as Generic Informational Inquiries
* **Customer Input**: `"I was charged $89.99 for an annual app I never purchased or downloaded! Refund my money immediately!"`
* **Predicted Intent**: `billing_subscription` (Confidence: 86.9%, Retrieval Sim: 0.773)
* **Actual System Decision**: `AUTO_HANDLE`
* **Expected Behavior**: `ESCALATE` (Reason: `DISPUTED_FINANCIAL_CHARGE`)
* **Actual Output**: *"To review charges and submit a refund request, visit our self-service portal: <URL> ^CD"*
* **Why It Happened**: While the intent `billing_subscription` was correctly predicted, high-dollar unauthorized charges require fraud investigation. The agent auto-handled by pointing to the standard self-service link.
* **Hypothesis**: Treating all `billing_subscription` inquiries uniformly causes under-escalation on contentious refund disputes.
* **Potential Fix**: Distinguish between read-only billing FAQs ("How to update card") and active disputes ("Charged for app I never purchased"), automatically escalating any charge dispute over $20.

---

### Failure 5: Semantic Retrieval Entity Mismatch on Gift Card Fraud
* **Customer Input**: `"Gift card says already redeemed but balance is zero!"`
* **Predicted Intent**: `billing_subscription` (Confidence: 79.4%, Retrieval Sim: 0.901)
* **Actual System Decision**: `AUTO_HANDLE`
* **Expected Behavior**: `ESCALATE` (Reason: `DISPUTED_FINANCIAL_CHARGE` / Gift Card Verification)
* **Actual Output**: *"Temporary authorization holds verify account validity and drop off in a few business days: <URL> ^IJ"*
* **Why It Happened**: The retrieval vector store achieved a high cosine similarity (0.901) with a billing precedent discussing authorization holds and balances, generating an entirely irrelevant explanation about pending holds.
* **Hypothesis**: Bi-encoder vector search without a cross-encoder reranker can produce high similarity scores between two sentences that share abstract financial terms ("balance", "zero", "account") even when the operational subject is completely different (gift cards vs bank holds).
* **Potential Fix**: Implement hybrid BM25 + dense retrieval with strict keyword filtering for physical card claims.
