# 12 — Reproducibility Guide & 15-Minute Benchmark Protocol

## 1. The 15-Minute Reproducibility Guarantee

A core deliverable of this assignment is that **any reviewer can clone this repository, run the evaluation pipeline on standard hardware, and reproduce headline benchmark results in under 15 minutes without paying for API keys**.

### Hardware & Environment Target
* **Operating System**: Windows 10/11, macOS, or Ubuntu Linux.
* **Hardware**: Standard 4-Core CPU, $\ge 8\text{ GB}$ RAM, no dedicated GPU required.
* **Python Runtime**: Python 3.10 to 3.12.
* **Network**: Required only for initial package download and HuggingFace weights download (`all-MiniLM-L6-v2` ~90MB).

---

## 2. Step-by-Step Reproduction Instructions

### Step 1: Environment Setup (~2 minutes)

```bash
# Clone the repository
git clone <REPO_URL>
cd hiver

# Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# Install free, open-source dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Environment Configuration (~30 seconds)

```bash
# Copy example configuration (defaults to local free execution)
cp .env.example .env
```

### Step 3: Data Preparation & Subsampling (~3 minutes)

```bash
# Downloads or extracts the deterministic 10k brand subset
python scripts/prepare_data.py --sample-size 10000 --brand AppleSupport
```

### Step 4: Train Baselines & Final Models (~3 minutes)

```bash
# Trains Baseline 1 (Majority), Baseline 2 (TF-IDF), and Final Embedding Classifier
# Builds FAISS retrieval index from the training split
python scripts/train_models.py
```

### Step 5: Run Automated Evaluation Harness (~4 minutes)

```bash
# Runs full evaluation on Holdout Test Set & 200-Example Golden Set
# Generates metrics JSON, confusion matrix plots, and baseline comparison table
python scripts/evaluate.py --run-judge
```

### Step 6: Inspect Results (~1 minute)

```bash
# View formatted headline metrics directly in terminal
python scripts/show_results.py
```

*Total Wall-Clock Time: ~13.5 minutes on a standard CPU machine.*

---

## 3. Seed & Determinism Control

All stochastic operations (dataset shuffling, train/test splitting, logistic regression solver initialization) are locked using a global random seed:

```python
RANDOM_SEED = 42
```

Configured centrally via `.env` and initialized across `numpy`, `torch`, and `scikit-learn` in `src/pipeline/__init__.py`.

---

## 4. Expected Artifact Locations

After execution, all verification artifacts are saved to deterministic paths:
* `results/metrics/intent_metrics.json`: Accuracy, Macro F1, Per-intent breakdown.
* `results/metrics/escalation_metrics.json`: Coverage, FAHR, Escalation Recall/Precision.
* `results/metrics/baseline_comparison.json`: Comparative table across all 3 systems.
* `results/metrics/judge_calibration.json`: Spearman correlation and Cohen's Kappa.
* `results/plots/confusion_matrix.png`: Normalized confusion matrix visual.
* `results/plots/coverage_vs_fahr.png`: Pareto frontier of coverage vs false auto-handles.
