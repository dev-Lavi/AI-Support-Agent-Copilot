# 09 — Golden Evaluation Set (200 Curated Examples)

## 1. Purpose & Curation Rationale

Evaluating customer support models exclusively on randomly split automated holdouts introduces blind spots: easy, repetitive queries dominate the metrics, while ambiguous, high-risk edge cases are underrepresented.

To rigorously audit the system, we construct a **200-example Golden Evaluation Set** hand-annotated with ground-truth intents, human escalation requirements, and reference resolutions.

---

## 2. Stratified Sampling Methodology

The 200 instances are selected via a multi-dimensional stratified sampling protocol rather than uniform random sampling:

1. **Intent Stratification (60% / 120 examples)**:
   * Proportional representation across all operational intents, with deliberate oversampling of minority classes (`feedback_complaint`, `repair_service_warranty`) to ensure statistical power.
2. **Linguistic & Length Diversity (20% / 40 examples)**:
   * **Short / Terse**: Very brief queries under 6 words (*"Battery dead again"*).
   * **Long / Multi-Sentence**: Complex narratives with multiple background details.
   * **Noisy / Informal**: Queries with heavy typos, phonetic spelling, all-caps, or emoji-only text (*"PLS FIX MY IPHONE CHARGER ISNT WERKNG 😭😭"*).
3. **Ambiguity & Boundary Hard Cases (10% / 20 examples)**:
   * Multi-intent queries touching both billing and app crashes (*"My game crashed and charged me twice for coins"*).
   * Vague or ungrounded queries (*"It's not working"*).
4. **Adversarial & High-Risk Cases (10% / 20 examples)**:
   * Sarcasm, severe customer distress, threats of legal action, account takeover panic.

---

## 3. Data Schema (`data/golden/golden_set.json`)

Each golden instance follows a structured schema:

```json
{
  "id": "gold_0142",
  "customer_message": "Got charged $14.99 for Apple Music today but I cancelled my subscription two weeks ago! Need my money back ASAP.",
  "conversation_id": "conv_98231",
  "gold_intent": "billing_subscription",
  "gold_should_escalate": true,
  "gold_escalation_reason": "DISPUTED_FINANCIAL_CHARGE",
  "requires_human_judgment": true,
  "difficulty_level": "medium",
  "key_entities": ["$14.99", "Apple Music", "cancelled subscription"],
  "gold_resolution_criteria": "Acknowledge the billing discrepancy, explain how to view subscription status at <URL>, and invite to DM to initiate billing investigation. Must NOT promise refund directly.",
  "reference_reply": "We understand how concerning unexpected charges can be. You can verify your cancellation status here: <URL>. Please DM us so we can look into this transaction directly.",
  "notes": "Direct refund request with emotional urgency; requires human review."
}
```

---

## 4. Human Annotation & Labelling Rubric

### Step 1: Assigning `gold_intent`
* Evaluator maps the query to the single most actionable operational intent.
* If two intents are present (e.g. device won't charge + store appointment), assign the primary root cause (`hardware_battery_power`) and note the secondary intent in `notes`.
* If completely unclassifiable, assign `other_unknown`.

### Step 2: Determining `gold_should_escalate`
* **Mark `true` (Must Escalate)** if:
  1. The issue requires access to private account records, credit cards, or internal CRM data.
  2. The customer expresses extreme anger, legal threats, or harassment.
  3. The query is too ambiguous to answer safely without guessing.
  4. The issue involves physical hardware safety (e.g., swollen battery, sparks).
* **Mark `false` (Auto-Handle Eligible)** if:
  1. The inquiry can be fully resolved with public troubleshooting steps, documented settings navigation, or official support links.

### Step 3: Drafting Reference Resolution Criteria
* Specifies what any acceptable agent reply must contain (e.g. link to battery settings) and must not contain (e.g. promise of free replacement).

---

## 5. Quality Assurance & Audit Process

* **Two-Pass Review**: All 200 examples undergo an audit pass to ensure boundary consistency between similar intents.
* **Leakage Verification**: Automated check confirms that zero `id` or exact text matches from `golden_set.json` exist in the training retrieval corpus.
