# 03 — Data Strategy, Cleaning & Leakage Prevention

## 1. Raw Dataset Characteristics

The primary dataset is **Customer Support on Twitter** (`thoughtvector/customer-support-on-twitter` on Kaggle), consisting of ~2.81 million tweets from 108 brands and their customers.

### Dataset Schema (`twcs.csv`)
* `tweet_id`: Unique integer identifier for each tweet.
* `author_id`: Anonymized identifier for the tweet author (brands use recognizable handles, e.g., `AppleSupport`, `AmazonHelp`, `Delta`).
* `inbound`: Boolean indicating whether the tweet was sent by a customer (`True`) or a brand agent (`False`).
* `created_at`: Tweet timestamp in RFC 2822 format (`Wed Oct 11 13:35:44 +0000 2017`).
* `text`: The raw text content of the tweet.
* `response_tweet_id`: Comma-separated list of tweet IDs that responded to this tweet.
* `in_response_to_tweet_id`: The ID of the parent tweet this tweet is directly answering.

---

## 2. Brand Selection Methodology

We do not choose a brand arbitrarily. We apply an empirical selection framework evaluating the dataset across five core criteria:

1. **Volume of Usable Customer-to-Brand Pairs**: Sufficient customer inbound messages that received a direct brand reply ($N \ge 10,000$ pairs).
2. **Intent Diversity**: Variety of distinct operational issues (e.g., billing, hardware, software, delivery, cancellations) rather than a single repetitive script.
3. **Resolution Actionability**: Historical brand replies that contain concrete guidance, troubleshooting steps, or explicit routing, rather than purely robotic "Please call us" responses.
4. **Thread Coherence**: Availability of clear parent-child links enabling clean first-turn pair extraction.
5. **Interview Explainability**: Domain familiarity for reviewers (e.g., Tech/Device Support like `@AppleSupport` or E-Commerce Support like `@AmazonHelp`).

### Candidate Brand Evaluation Matrix (Hypothesis & Analysis Framework)

| Candidate Brand | Domain | Est. Volume | Pros | Cons / Challenges |
| :--- | :--- | :--- | :--- | :--- |
| **`@AppleSupport`** | Consumer Tech & OS | Very High (~100k+) | Highly structured, technical issues, rich troubleshooting steps. | Heavy usage of DM links and apple.co support articles. |
| **`@AmazonHelp`** | E-Commerce & Delivery | Extremely High (~150k+) | Massive volume, diverse issues (orders, delivery, refunds). | High repetition of generic redirection templates. |
| **`@Delta` / `@British_Airways`**| Travel & Airlines | High (~40k+) | High-stakes issues (delays, baggage, cancellations, seats). | Time-sensitive context dependent on external flight schedules. |
| **`@SpotifyCares`** | Subscription & Streaming | Moderate (~30k+) | Clean digital service issues (login, playlists, premium billing). | Narrower intent variety compared to Apple or Amazon. |

*Note: The script `scripts/analyze_brands.py` is provided to compute empirical distributions and confirm the optimal brand choice before finalizing the training pipeline.*

---

## 3. Data Processing & Pair Reconstruction Pipeline

```text
Raw Twitter CSV (twcs.csv)
            │
            ▼
[Filter Brand & Inbound/Outbound]
  - Retain tweets authored by Brand or referencing Brand
            │
            ▼
[Thread Reconstruction]
  - Map `in_response_to_tweet_id` to reconstruct customer -> brand pairs
  - Isolate initial customer inquiry (first-turn inbound) paired with brand's initial response
            │
            ▼
[Text Cleaning & Normalization]
  - Mask sensitive handles (@AppleSupport -> <BRAND>, @user123 -> <USER>)
  - Normalize URLs (http://t.co/... -> <URL>)
  - Unescape HTML entities (&amp; -> &, &gt; -> >)
  - Normalize unicode & redundant whitespace
            │
            ▼
[Deduplication & Quality Filtering]
  - Remove empty, truncated, or purely emoji tweets
  - Deduplicate identical automated bot spam queries
            │
            ▼
[Conversation-Level Grouped Splitting]
  - Group by conversation thread ID
  - Stratify into Train (70%), Validation (15%), Test (15%)
            │
            ▼
[Golden Evaluation Set Sampling]
  - Stratified extraction of 200 diverse, ambiguous, and edge-case instances
```

---

## 4. Strict Data Leakage Prevention Protocols

Data leakage is the most common cause of inflated, untrustworthy performance in conversational AI benchmarks. We enforce five non-negotiable isolation rules:

### Rule 1: Conversation-Level Grouping
Tweets belonging to the same conversation thread (or same customer inquiry session) **must never be split between Train and Test**. All turns of a thread are assigned strictly to one split.

### Rule 2: Strict Retrieval Index Isolation
The FAISS retrieval vector store is populated **exclusively from the Training Split**.
* $\text{Index Corpus} = \mathcal{D}_{\text{train}}$
* $\mathcal{D}_{\text{val}} \cap \text{Index Corpus} = \emptyset$
* $\mathcal{D}_{\text{test}} \cap \text{Index Corpus} = \emptyset$
* $\mathcal{D}_{\text{golden}} \cap \text{Index Corpus} = \emptyset$
If a test query is evaluated, the system can never retrieve its own ground-truth historical reply.

### Rule 3: Exact & Near-Duplicate Removal
Customers often tweet standard templates. We compute MinHash / exact text hashes to ensure that exact duplicate tweets appearing in the test set do not artificially inflate retrieval or classification accuracy.

### Rule 4: Temporal Integrity
Where chronological metadata is available, splits are created to reflect real-world deployment: training on past conversations, evaluating on future conversations.

### Rule 5: Golden Set Complete Independence
The 200 hand-labelled Golden Set examples are held out entirely and never touched during hyperparameter tuning, embedding model selection, or classifier optimization.

---

## 5. Subsampling Strategy for 15-Minute Reproducibility

Loading and indexing the full ~3M row CSV takes substantial RAM and >30 minutes on standard hardware. To guarantee our **<15-minute reproduction promise**:
* We extract a clean, deterministic subset of $N = 10,000$ high-quality customer-brand pairs for the selected brand.
* The extracted dataset is saved as lightweight Parquet / JSON (`data/processed/brand_pairs.parquet`).
* Full raw data processing scripts are preserved in `scripts/prepare_data.py` for auditability, but evaluation scripts default to the processed reproducible subset.
