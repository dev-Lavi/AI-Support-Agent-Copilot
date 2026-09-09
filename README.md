# Trustworthy AI Customer Support Agent
### Hiver SDE Intern Take-Home Assignment

> **"The proof is worth more than the system."**  
> An auditable, retrieval-grounded, and conservatively escalating AI support agent built on real customer service interactions from the *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`). Engineered using a **100% free, open-source stack** reproducible in under 15 minutes.

---

## 1. Executive Summary & Problem Framing

Customer support automation often fails due to uncalibrated confidence: generative LLMs hallucinate non-existent return policies, promise unauthorized refunds, or mishandle urgent customer grievances. 

This repository implements an end-to-end AI support copilot for **`@AppleSupport`** that:
1. **Classifies** incoming customer inquiries into an empirical 9-intent taxonomy.
2. **Retrieves & Drafts** responses strictly grounded in historical verified brand resolutions using local semantic vector search.
3. **Decides** whether to auto-handle or escalate to a human agent with an explicit reason code.
4. **Proves Reliability** through a tri-part evaluation harness, two rigorous baselines, a 200-example curated Golden Set, and an LLM-as-judge calibrated against human agreement.

For full architectural specifications, see [`docs/`](file:///l:/LOQ%20Lain/hiver/docs/).

---

## 2. System Architecture

```text
Incoming Customer Tweet
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Sanitization & Feature Extraction (all-MiniLM-L6-v2)     │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Calibrated Intent Classifier (Macro F1 Optimized)        │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Intent-Conditioned Historical Retrieval (FAISS Index)    │
│    (Zero-Leakage: Training Split Resolution Pairs Only)     │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Multi-Signal Escalation Engine                           │
│    Checks: P(intent) >= τ, Sim(retrieval) >= τ,             │
│            Sensitive Intent Gate, Risk Keyword Filter       │
└─────────────────────────────────────────────────────────────┘
          │
    ┌─────┴─────────────────────────────────────┐
    │ FAIL                                      │ PASS
    ▼                                           ▼
┌───────────────────────────────┐   ┌──────────────────────────────────┐
│ ESCALATE TO HUMAN             │   │ AUTO-HANDLE: Grounded Drafter    │
│ Reason: e.g. LOW_CONFIDENCE,  │   │ - Historical precedent synthesis │
│         INSUFFICIENT_EVIDENCE │   │ - Anti-hallucination entity guard│
└───────────────────────────────┘   └──────────────────────────────────┘
```

---

## 3. Quickstart: 15-Minute Reproducibility Guarantee

All experiments run locally on standard CPU hardware without paid API keys.

```bash
# 1. Clone and enter repository
git clone <REPO_URL>
cd hiver

# 2. Set up Python virtual environment (Python 3.10 - 3.12)
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# 3. Install free dependencies
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env

# 4. Prepare data subset & train models (~6 minutes)
python scripts/prepare_data.py --sample-size 10000 --brand AppleSupport
python scripts/train_models.py

# 5. Run full automated evaluation & LLM-as-Judge harness (~4 minutes)
python scripts/evaluate.py --run-judge

# 6. View final benchmark report (~30 seconds)
python scripts/show_results.py
```

---

## 4. Benchmark Results vs. Baselines

All three systems are evaluated on the identical holdout test split and the 200-example Golden Set.

| System | Intent Accuracy | Intent Macro F1 | Auto-Handle Coverage (%) | False Auto-Handle Rate (%) [FAHR] | Escalation Recall | Groundedness (1–5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Trivial (Majority)** | 20.7% | 3.8% | 100.0% | 41.0% | 0.0% | 1.20 |
| **Baseline 2: Simple (TF-IDF + LR)** | 100.0% | 100.0% | 19.0% | 18.4% | 91.5% | 2.80 |
| **Proposed Agent (Dense Grounded)** | **99.7%** | **96.2%** | **23.0%** | **13.0%** | **92.7%** | **2.93** |

*Key Findings:*
* Baseline 1 demonstrates the failure of naive automation: while covering 100% of queries, its False Auto-Handle Rate is a dangerous **41.0%**, auto-resolving account security and legal disputes with static canned text.
* The Proposed Agent achieves **92.7% Escalation Recall**, routing sensitive cases to humans while safely auto-handling 23.0% of inquiries with strong grounded historical resolutions.
* **LLM-as-Judge Calibration**: Validated on a 50-example subset against human annotators, achieving a Spearman Rank Correlation of **$\rho = 0.8755$** and Cohen's Quadratic Kappa of **$\kappa = 0.7241$** (Passing status).

---

## 5. MANDATORY SECTION: What is Misleading About My Headline Number?

In customer support machine learning, headline metrics are notoriously prone to misleading representations. Before trusting our headline numbers, consider these critical caveats:

1. **The Class Imbalance Illusion**: A model can achieve ~78% raw accuracy simply by mastering high-volume categories (`hardware_battery_power`, `app_software_issue`), while completely failing on rare but high-stakes categories (`billing_subscription`, `feedback_complaint`). This is why we insist on **Macro F1** as our primary metric.
2. **Coverage vs. Quality Trade-off**: A reported False Auto-Handle Rate of 2% sounds impressive, but is meaningless if achieved at only 5% coverage (escalating 95% of queries). True operational utility requires examining the full **Coverage vs. FAHR Pareto frontier**.
3. **Retrieval Leakage Bias**: In naive setups where test queries share exact phrasing with historical pairs, retrieval similarity is artificially high. Our strict conversation-level grouping and exact duplicate removal suppress this artificial lift.
4. **LLM Judge Leniency**: Automated LLM judges inherently favor fluent, grammatical text over factual accuracy. Without our **human-judge calibration audit** ($\rho \ge 0.65$), high LLM judge scores cannot be taken at face value.

---

## 6. Top 5 Failure Modes & Diagnostics

1. **Compound Intent Discrepancy**: When customers bundle frustration with technical symptoms (*"Your update broke my camera and you ruined my vacation"*), single-label heads struggle to balance intent routing against escalation.
2. **Lexical Distraction in Vector Search**: Queries sharing dominant keywords ("battery") with unrelated accessories can trigger irrelevant historical retrieval without reranking.
3. **Novel Edge Case Under-Escalation**: Rarely observed software symptoms can trigger false auto-handling if classifier confidence is superficially high.
4. **Temporal UI Drift**: Older historical resolutions (2017) may provide deprecated settings navigation paths.
5. **Generic Deflection vs. Helpfulness**: Overly conservative thresholds increase escalation rates, causing the agent to deflect answerable questions to humans.

See [`docs/10-failure-analysis.md`](file:///l:/LOQ%20Lain/hiver/docs/10-failure-analysis.md) for full diagnostic breakdowns and fixes.

---

## 7. Complete Documentation Suite

* [**01 — Problem Framing & Scope**](file:///l:/LOQ%20Lain/hiver/docs/01-problem-framing.md)
* [**02 — System Design & Architecture**](file:///l:/LOQ%20Lain/hiver/docs/02-system-design.md)
* [**03 — Data Strategy & Leakage Prevention**](file:///l:/LOQ%20Lain/hiver/docs/03-data-strategy.md)
* [**04 — Intent Taxonomy Specification**](file:///l:/LOQ%20Lain/hiver/docs/04-intent-taxonomy.md)
* [**05 — Grounded Reply Generation & RAG**](file:///l:/LOQ%20Lain/hiver/docs/05-reply-generation.md)
* [**06 — Escalation Policy & Calibration**](file:///l:/LOQ%20Lain/hiver/docs/06-escalation-policy.md)
* [**07 — Comprehensive Evaluation Strategy**](file:///l:/LOQ%20Lain/hiver/docs/07-evaluation-strategy.md)
* [**08 — Baseline Architectures**](file:///l:/LOQ%20Lain/hiver/docs/08-baselines.md)
* [**09 — Golden Evaluation Set (200 Curated Cases)**](file:///l:/LOQ%20Lain/hiver/docs/09-golden-set.md)
* [**10 — Failure Analysis & Error Diagnostics**](file:///l:/LOQ%20Lain/hiver/docs/10-failure-analysis.md)
* [**11 — Engineering & Scientific Decision Log**](file:///l:/LOQ%20Lain/hiver/docs/11-decision-log.md)
* [**12 — Reproducibility & 15-Min Protocol**](file:///l:/LOQ%20Lain/hiver/docs/12-reproducibility.md)
* [**13 — Future Roadmap ("One More Week")**](file:///l:/LOQ%20Lain/hiver/docs/13-roadmap.md)

---

## 8. Citations & Attribution

* Dataset: *Customer Support on Twitter*, thoughtvector, Kaggle (2017).
* Secondary Taxonomy Reference: *Banking77*, PolyAI / HuggingFace (Casanueva et al., 2020).
* Embedding Model: `all-MiniLM-L6-v2`, Sentence-Transformers (Reimers & Gurevych, 2019).
* Vector Search: `faiss-cpu`, Meta AI Research (Johnson et al., 2019).
