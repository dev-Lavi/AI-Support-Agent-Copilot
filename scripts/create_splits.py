"""Script to perform conversation-grouped, leakage-safe train/val/test splits.

Usage:
    python scripts/create_splits.py --input data/processed/brand_pairs.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Create leakage-safe dataset splits.")
    parser.add_argument("--input", type=str, default="data/processed/brand_pairs.parquet", help="Path to processed pairs")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for splits")
    args = parser.parse_args()

    print(f"[create_splits] Splitting dataset from {args.input} with strict thread grouping...")
    print("[create_splits] Pipeline skeleton initialized. Awaiting Phase 2 implementation approval.")


if __name__ == "__main__":
    main()
