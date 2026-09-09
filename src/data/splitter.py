"""Conversation-level grouped splitting module ensuring strict zero-leakage."""

from typing import Tuple
import numpy as np
import pandas as pd


def create_grouped_splits(
    df: pd.DataFrame,
    group_col: str = "conversation_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into Train, Validation, and Test sets grouped strictly by `group_col`

    to prevent cross-turn thread contamination and data leakage.

    Args:
        df: DataFrame containing the conversation pairs.
        group_col: Column name identifying thread/conversation groups.
        train_ratio: Proportion for training (default 0.70).
        val_ratio: Proportion for validation (default 0.15).
        test_ratio: Proportion for testing (default 0.15).
        random_seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    assert group_col in df.columns, f"Group column '{group_col}' missing from DataFrame"

    rng = np.random.RandomState(random_seed)
    unique_groups = list(df[group_col].unique())
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    train_df = df[df[group_col].isin(train_groups)].copy().reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].copy().reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].copy().reset_index(drop=True)

    # Zero-leakage verification assertions
    train_ids = set(train_df[group_col])
    val_ids = set(val_df[group_col])
    test_ids = set(test_df[group_col])

    assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected: Train and Val share groups!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected: Train and Test share groups!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage detected: Val and Test share groups!"

    return train_df, val_df, test_df
