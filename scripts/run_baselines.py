"""Script to train and evaluate Baseline 1 (Majority) and Baseline 2 (TF-IDF + LogReg).

Usage:
    python scripts/run_baselines.py --test-path data/splits/test.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for baseline models.")
    parser.add_argument("--test-path", type=str, default="data/splits/test.parquet", help="Path to holdout test split")
    args = parser.parse_args()

    print("[run_baselines] Evaluating Baseline 1: Majority Class...")
    print("[run_baselines] Evaluating Baseline 2: TF-IDF + Logistic Regression...")
    print("[run_baselines] Pipeline skeleton initialized. Awaiting Phase 2 implementation approval.")


if __name__ == "__main__":
    main()
