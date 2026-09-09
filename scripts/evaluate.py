"""Master Evaluation Script: Computes metrics across baselines, golden set, and judge calibration.

Usage:
    python scripts/evaluate.py --run-judge
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.pipeline.agent import SupportAgentPipeline
from src.evaluation.harness import EvaluationHarness
from src.intents.taxonomy import INTENTS


def plot_confusion_matrix(cm_list: list, output_path: Path):
    """Plots and saves normalized confusion matrix heatmap."""
    cm = np.array(cm_list)
    # Normalize rows
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    short_labels = [
        "battery", "app_sw", "account", "update", "billing",
        "network", "repair", "complaint", "unknown"
    ]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar=True
    )
    plt.title("Normalized Intent Classification Confusion Matrix (Agent vs Test Set)")
    plt.xlabel("Predicted Intent")
    plt.ylabel("Ground Truth Intent")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[eval] Confusion matrix visual saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate support agent across all criteria.")
    parser.add_argument("--test-path", type=str, default="data/splits/test.csv", help="Path to holdout test set")
    parser.add_argument("--golden-set", type=str, default="data/golden/golden_set.json", help="Path to golden set")
    parser.add_argument("--models-dir", type=str, default="results/models", help="Path to trained models")
    parser.add_argument("--run-judge", action="store_true", help="Run LLM-as-judge calibration")
    args = parser.parse_args()

    train_path = PROJECT_ROOT / "data/splits/train.csv"
    if not train_path.exists():
        print("[eval] Train split missing. Running data preparation first...")
        from scripts.prepare_data import main as run_prepare
        run_prepare()

    train_df = pd.read_csv(train_path)

    # Check if models exist, if not train them
    models_dir = PROJECT_ROOT / args.models_dir
    if not (models_dir / "intent_classifier.pkl").exists():
        print("[eval] Trained models not found. Running training first...")
        from scripts.train_models import main as run_train
        run_train()

    print("[eval] Loading trained support agent pipeline...")
    pipeline = SupportAgentPipeline.load(str(models_dir))

    harness = EvaluationHarness(
        test_path=str(PROJECT_ROOT / args.test_path),
        golden_path=str(PROJECT_ROOT / args.golden_set),
        results_dir=str(PROJECT_ROOT / "results/metrics")
    )

    results = harness.run_full_benchmark(train_df=train_df, pipeline=pipeline)

    # Generate Confusion Matrix Plot
    cm_data = results["agent_intent"]["confusion_matrix"]
    plot_confusion_matrix(cm_data, PROJECT_ROOT / "results/plots/confusion_matrix.png")


if __name__ == "__main__":
    main()
