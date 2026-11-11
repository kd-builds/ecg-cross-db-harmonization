"""
Patient-disjoint and provenance-aware dataset partitioning.
Corresponds to Section 6 (Data partitioning) of the research proposal.
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def create_patient_disjoint_splits(
    manifest_df: pd.DataFrame,
    group_col: str = "patient_id",
    train_size: float = 0.70,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions a dataset into patient-disjoint train, validation, and internal-test sets.
    Verifies zero group overlap across all three partitions.
    """
    if group_col not in manifest_df.columns or manifest_df[group_col].isna().any():
        raise ValueError(f"Column '{group_col}' must be present and non-null for patient-disjoint splitting.")

    gss_outer = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, temp_idx = next(gss_outer.split(manifest_df, groups=manifest_df[group_col]))

    train_df = manifest_df.iloc[train_idx].copy()
    temp_df = manifest_df.iloc[temp_idx].copy()

    relative_val_size = val_size / (1.0 - train_size)
    gss_inner = GroupShuffleSplit(n_splits=1, train_size=relative_val_size, random_state=random_state)
    val_idx, test_idx = next(gss_inner.split(temp_df, groups=temp_df[group_col]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Audit zero leakage
    train_groups = set(train_df[group_col])
    val_groups = set(val_df[group_col])
    test_groups = set(test_df[group_col])
    assert train_groups.isdisjoint(val_groups), "Leakage detected between train and validation sets!"
    assert train_groups.isdisjoint(test_groups), "Leakage detected between train and internal test sets!"
    assert val_groups.isdisjoint(test_groups), "Leakage detected between validation and internal test sets!"

    return train_df, val_df, test_df
