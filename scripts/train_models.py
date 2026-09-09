"""Script to train intent classifiers, build the FAISS retrieval index,

and calibrate escalation policy thresholds.

Usage:
    python scripts/train_models.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.pipeline.agent import SupportAgentPipeline


def main():
    print("=" * 70)
    print("TRAINING AI CUSTOMER SUPPORT AGENT")
    print("=" * 70)

    train_path = PROJECT_ROOT / "data/splits/train.csv"
    val_path = PROJECT_ROOT / "data/splits/val.csv"

    if not train_path.exists():
        print(f"[train] Training data not found at {train_path}. Running prepare_data.py first...")
        from scripts.prepare_data import main as run_prepare
        run_prepare()

    print(f"[train] Loading training split: {train_path}")
    train_df = pd.read_csv(train_path)

    val_df = None
    if val_path.exists():
        print(f"[train] Loading validation split for calibration: {val_path}")
        val_df = pd.read_csv(val_path)

    # Initialize and train pipeline
    pipeline = SupportAgentPipeline(brand_handle="AppleSupport")
    pipeline.fit(train_df, val_df)

    # Save trained pipeline artifacts
    model_dir = PROJECT_ROOT / "results/models"
    print(f"[train] Saving trained model and retrieval index to {model_dir}/")
    pipeline.save(str(model_dir))

    print("=" * 70)
    print("MODEL TRAINING & FAISS INDEXING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
