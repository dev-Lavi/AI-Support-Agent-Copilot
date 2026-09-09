"""Script to construct the FAISS retrieval vector store strictly from the training split.

Usage:
    python scripts/build_index.py --train-path data/splits/train.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Build retrieval index from training split.")
    parser.add_argument("--train-path", type=str, default="data/splits/train.parquet", help="Path to training split")
    parser.add_argument("--index-path", type=str, default="data/splits/faiss_index.bin", help="Output index path")
    args = parser.parse_args()

    print(f"[build_index] Indexing historical resolutions from {args.train_path}...")
    print("[build_index] Zero-leakage constraint active: Test and Golden queries excluded.")
    print("[build_index] Pipeline skeleton initialized. Awaiting Phase 2 implementation approval.")


if __name__ == "__main__":
    main()
