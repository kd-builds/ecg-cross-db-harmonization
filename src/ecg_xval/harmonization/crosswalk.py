"""
Versioned label crosswalk and per-class eligibility/masking engine.
Corresponds to Section 5 (Label Harmonization) & Weeks 4-5 of the research plan.
"""
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


class LabelCrosswalk:
    """
    Maps source database diagnostic codes to canonical shared rhythm classes
    while preserving co-occurring labels and generating explicit evaluation masks.
    """

    def __init__(self, crosswalk_csv: Path, mapping_tier: str = "primary"):
        self.df = pd.read_csv(crosswalk_csv)
        self.mapping_tier = mapping_tier
        self._build_lookup()

    def _build_lookup(self) -> None:
        active = self.df[self.df["inclusion_status"] == "included"]
        if self.mapping_tier == "primary":
            active = active[active["mapping_tier"] == "primary"]
        self.canonical_classes: List[str] = sorted(active["canonical_class"].unique().tolist())
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.canonical_classes)}

    def encode_record_labels(
        self,
        source_database: str,
        raw_codes: List[str],
        annotated_universe: Set[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            y_true: binary target vector of shape (num_classes,)
            eval_mask: binary eligibility mask of shape (num_classes,) where 1 = evaluable,
                       0 = masked (e.g., class not annotated in source database or uncertain).
        """
        num_classes = len(self.canonical_classes)
        y_true = np.zeros(num_classes, dtype=np.float32)
        eval_mask = np.ones(num_classes, dtype=np.float32)

        db_rows = self.df[self.df["source_database"] == source_database]
        code_map = dict(zip(db_rows["original_code"].astype(str), db_rows["canonical_class"]))

        for code in raw_codes:
            canonical = code_map.get(str(code))
            if canonical in self.class_to_idx:
                y_true[self.class_to_idx[canonical]] = 1.0

        # Mask classes that are not part of this database's audited annotation universe
        for cls_name, idx in self.class_to_idx.items():
            if cls_name not in annotated_universe:
                eval_mask[idx] = 0.0

        return y_true, eval_mask
