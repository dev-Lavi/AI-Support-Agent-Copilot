# 02 — End-to-End System Design & Architecture

## 1. Architectural Overview

The system processes incoming customer messages through an auditable, six-stage pipeline designed for deterministic reproducibility and zero paid API dependencies.

```text
Incoming Customer Tweet
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Ingestion, Sanitization & Context Preparation      │
│ - URL/mention normalization, emoji parsing, thread context  │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Intent Classification & Posterior Calibration      │
│ - Dense Sentence-Transformer embedding (all-MiniLM-L6-v2)   │
│ - Calibrated probability distribution over N defined intents│
│ - Outputs: Predicted Intent ŷ, Intent Confidence P(ŷ|x)     │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Intent-Conditioned Resolution Retrieval (FAISS)    │
│ - Filter historical corpus by predicted intent (or global)  │
│ - Retrieve Top-K nearest historical customer-brand pairs    │
│ - Compute Max Cosine Similarity Score S_retrieval           │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Multi-Signal Escalation Decision Engine            │
│ Evaluates:                                                  │
│ 1. P(ŷ|x) < τ_intent ?                                      │
│ 2. S_retrieval < τ_retrieval ?                              │
│ 3. Is intent marked sensitive (e.g. fraud, account loss)?   │
│ 4. Are retrieved historical resolutions conflicting?        │
│ 5. Is user message ungrounded / missing required context?   │
└─────────────────────────────────────────────────────────────┘
          │
    ┌─────┴─────────────────────────────────────┐
    │                                           │
    ▼ [ESCALATE TRIGGERED]                      ▼ [ELIGIBLE FOR AUTO-HANDLE]
┌───────────────────────────────┐   ┌──────────────────────────────────┐
│ Generate Escalation Object:   │   │ Stage 5: Grounded Reply Drafter  │
│ - Decision: "ESCALATE"        │   │ - Retrieval-augmented prompt     │
│ - Reason Code: e.g.           │   │ - Local HF model / template      │
│   LOW_CONFIDENCE              │   │ - Strict entity & policy guard   │
│ - Recommended Human Priority  │   │ - Decision: "AUTO_HANDLE"        │
└───────────────────────────────┘   └──────────────────────────────────┘
                │                                    │
                └─────────────────┬──────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: Audit Logging & Evaluation Harness                 │
│ - Structured JSON output for inspection & evaluation        │
│ - Automated metric recording                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications & Interfaces

### Stage 1: Text Sanitization & Ingestion
* **Module**: `src/data/cleaner.py`
* **Input**: Raw tweet string + conversation thread metadata.
* **Operations**:
  * Strips handle prefixes (`@Brand`, `@customer`) while preserving structural hashtags and references.
  * Replaces raw URLs with tokens (`<URL>`) to prevent models overfitting to dynamic link parameters.
  * Normalizes whitespace, emoji representations, and casing where appropriate.
* **Output**: Cleaned normalized string $x \in \mathcal{X}$.

### Stage 2: Intent Classification
* **Module**: `src/intents/embedding_classifier.py`
* **Input**: Cleaned string $x$.
* **Mechanism**:
  * Feature Extraction: Sentence-Transformer (`sentence-transformers/all-MiniLM-L6-v2`), producing 384-dimensional dense semantic vectors.
  * Classification Head: Softmax-calibrated Logistic Regression or Linear SVM with Platt scaling.
* **Output**:
  * $\hat{y} = \arg\max_{k} P(y_k | x)$
  * Confidence score $c_{\text{intent}} = \max_k P(y_k | x) \in [0, 1]$
  * Complete posterior vector across all candidate classes.

### Stage 3: Semantic Resolution Retrieval
* **Module**: `src/retrieval/index.py`
* **Input**: Cleaned string $x$, predicted intent $\hat{y}$.
* **Index**: FAISS IndexFlatIP (Inner Product over L2-normalized embeddings) indexed over historical training-set customer queries.
* **Metadata Store**: Associated historical brand replies, resolution actions, conversation IDs, and timestamps.
* **Zero-Leakage Constraint**: The FAISS index is constructed **strictly from the Training split**. Test and Golden set queries/replies are isolated and strictly excluded.
* **Output**: Top-$k$ evidence tuples:
  $$\mathcal{E} = \{ (q_i^{\text{hist}}, r_i^{\text{hist}}, \text{sim}_i) \}_{i=1}^k$$
  where $\text{sim}_1$ represents the maximum retrieval similarity $S_{\text{retrieval}}$.

### Stage 4: Multi-Signal Escalation Decision Engine
* **Module**: `src/escalation/policy.py`
* **Inputs**: $\hat{y}$, $c_{\text{intent}}$, $\mathcal{E}$, $S_{\text{retrieval}}$, and raw text $x$.
* **Logic**: Evaluates a priority rule chain against validation-tuned thresholds $(\tau_{\text{intent}}, \tau_{\text{retrieval}})$.
* **Output**:
  ```json
  {
    "decision": "AUTO_HANDLE" | "ESCALATE",
    "reason_code": "LOW_INTENT_CONFIDENCE" | "INSUFFICIENT_HISTORICAL_EVIDENCE" | "SENSITIVE_CASE" | "UNKNOWN_INTENT" | "SUCCESSFUL_MATCH",
    "confidence_scores": {
      "intent_confidence": 0.84,
      "retrieval_similarity": 0.76
    }
  }
  ```

### Stage 5: Grounded Reply Generation
* **Module**: `src/generation/grounded_drafter.py`
* **Inputs**: Clean query $x$, top retrieved evidence $\mathcal{E}$, predicted intent $\hat{y}$.
* **Mechanism**:
  * RAG Synthesis: Local open-source model (`google/flan-t5-base` or lightweight quantized model) or structured template synthesis parameterized by retrieved historical resolutions.
  * Grounding Rule: The drafter may only extract verified actions, contact channels, and instructions present in $\mathcal{E}$.
* **Output**: Draft reply string $\hat{r}$.

### Stage 6: Evaluation Harness & LLM Judge
* **Module**: `src/evaluation/harness.py`, `src/evaluation/judge.py`
* **Evaluates**:
  * Intent metrics (Accuracy, Macro F1, Weighted F1, Confusion Matrix).
  * Escalation metrics (Precision, Recall, Coverage, False Auto-Handle Rate).
  * Reply Quality via calibrated LLM-as-judge compared against human annotations.

---

## 3. Technology Stack & Free/Local Justification

| Layer | Selected Tool | Alternative Considered | Justification for Selection |
| :--- | :--- | :--- | :--- |
| **Data Processing** | `pandas`, `numpy` | `polars`, `duckdb` | Maximum portability, universal standard, zero compilation issues on Windows/Linux. |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | `OpenAI text-embedding-3-small` | 100% Free, runs locally on CPU in milliseconds, 384 dims, highly competitive semantic performance. |
| **Vector Index** | `faiss-cpu` (with `sklearn.metrics.pairwise` fallback) | `Pinecone`, `Qdrant`, `Chroma` | Local in-memory index, zero network overhead, no credentials, sub-millisecond query time. |
| **Classifiers** | `scikit-learn` | `PyTorch` custom MLP | Scikit-learn Logistic Regression provides exact probability calibration, rapid training (<30s), interpretable coefficients. |
| **Generation** | Local HuggingFace / Grounded Template Synthesizer | `OpenAI GPT-4o-mini`, `Claude 3.5 Haiku` | Guarantees 100% free offline reproducibility without rate limits or API key requirements. |
| **Storage** | Parquet / JSON / CSV | SQLite / Postgres | File-based storage is native to Git and enables instant pipeline reproducibility. |
