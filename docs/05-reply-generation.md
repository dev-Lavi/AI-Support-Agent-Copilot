# 05 — Grounded Reply Generation & Retrieval-Augmentation

## 1. Grounding Philosophy

In automated customer support, an ungrounded LLM is a liability. Left unconstrained, generative models hallucinate non-existent warranty policies, invent toll-free phone numbers, promise financial refunds, or create broken documentation links.

Our reply generation system operates under a strict principle:
> **The generator is an evidence-conditioned synthesizer, not an autonomous policy maker.**

If historical support data does not provide direct, verified precedent for resolving a customer's query, the agent must **escalate to a human agent** rather than guessing.

---

## 2. RAG Retrieval Architecture

```text
Incoming Customer Query (x) + Predicted Intent (ŷ)
                      │
                      ▼
       [Intent-Filtered Semantic Search]
                      │
   FAISS Index (Embeddings of Historical Training Queries)
                      │
                      ▼
   Top-K Historical Candidates:
   [ (q_1, r_1, sim_1), ..., (q_k, r_k, sim_k) ]
                      │
                      ▼
          [Evidence Quality Filter]
   - Is max(sim) >= τ_retrieval (e.g. 0.65)?
   - Do top candidates agree on the resolution path?
          │
    ┌─────┴─────────────────────────┐
    │ FAIL                          │ PASS
    ▼                               ▼
[Trigger Escalation]        [Construct Grounded Context]
Reason: INSUFFICIENT_EVIDENCE       │
                                    ▼
                        [Grounded Prompt Synthesis]
                                    │
                                    ▼
                     [Response Safety & Entity Guard]
                                    │
                                    ▼
                             Draft Reply (r̂)
```

---

## 3. Retrieval Index Data Schema

Each document in the retrieval index (`data/splits/retrieval_corpus.json`) represents an authentic historical resolution pair extracted strictly from the **Training Split**:

```json
{
  "doc_id": "hist_pair_08421",
  "historical_query": "My iPhone battery dies within 2 hours after updating to the latest iOS.",
  "historical_reply": "We want to help ensure your battery is performing at its best. Check out your Battery Health under Settings > Battery, and let us know what percentage you see: <URL>",
  "intent": "hardware_battery_power",
  "conversation_id": "conv_92384",
  "brand_handle": "AppleSupport",
  "timestamp": "2017-10-12T14:22:00Z"
}
```

---

## 4. Prompt Engineering for Grounded Synthesis

When using a local open-source transformer (`google/flan-t5-base` or quantized model), the prompt strictly enforces evidence containment:

```text
[SYSTEM INSTRUCTION]
You are a professional customer support assistant for AppleSupport on Twitter.
Your replies must be grounded EXCLUSIVELY in the provided Historical Resolutions.
Do not invent policies, phone numbers, refund amounts, or URLs not found in the context.
If the historical evidence does not contain a direct resolution, output "ESCALATE_TO_HUMAN".
Keep responses concise, polite, empathetic, and under 240 characters.

[HISTORICAL RESOLUTIONS]
Case 1 (Similarity: 0.82):
Customer: "iPhone screen is black but it vibrates when plugged in."
Brand Reply: "Let's try a force restart on your iPhone. Follow the steps outlined here: <URL> and let us know if the Apple logo appears."

Case 2 (Similarity: 0.78):
Customer: "Screen won't light up but notifications make sound."
Brand Reply: "We're here to help. Have you tried a hard reset yet? You can follow the guide here: <URL>"

[INCOMING CUSTOMER QUERY]
Customer: "My phone vibrates and rings but screen stays totally pitch black."

[GROUNDED DRAFT REPLY]
```

---

## 5. Hallucination Safeguards & Verification

Every generated reply $\hat{r}$ passes through an automated post-generation guardrail before being approved for auto-handling:

1. **Entity Verification**: Any URL, email, or telephone number appearing in $\hat{r}$ must have an exact character match in the retrieved evidence set $\mathcal{E}$. Fabricated URLs trigger an immediate escalation.
2. **Policy Claim Guard**: Keyword patterns implying financial commitments (`"I have refunded"`, `"We will credit your account"`, `"Replacement is free of charge"`) trigger immediate escalation unless explicitly present in historical precedent.
3. **Contradiction Detection**: If Case 1 advises `"Hard restart your device"` and Case 2 advises `"Bring device to service center immediately"` with comparable similarity, the system flags `CONFLICTING_EVIDENCE` and escalates.
4. **Length & Tone Bounds**: Checks for appropriate character limits (Twitter standard $\le 280$ characters) and absence of hostile or apologetically excessive phrasing.
