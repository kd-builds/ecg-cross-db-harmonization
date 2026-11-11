"""
Sensitivity and error analysis across label mappings (conservative vs. broad),
signal quality strata, and class imbalance.
Corresponds to Section 7 & Weeks 13-14 of the research proposal.
"""
import pandas as pd


def compute_transfer_gap_table(within_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """Quantifies internal-to-external macro-F1 change across database pairs."""
    merged = external_df.merge(
        within_df[["target_database", "macro_f1"]].rename(columns={"macro_f1": "within_db_macro_f1"}),
        on="target_database",
        how="left",
    )
    merged["delta_macro_f1"] = merged["external_macro_f1"] - merged["within_db_macro_f1"]
    return merged
