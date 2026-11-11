"""
Dataset provenance, duplicate record detection, and annotation coverage auditing.
Corresponds to Section 4 (Scope and Data) & Weeks 1-3 of the research plan.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import hashlib
import numpy as np
import pandas as pd


@dataclass
class CohortAuditSummary:
    cohort_name: str
    provenance_group: str
    total_records: int
    unique_patients: Optional[int]
    usable_12_lead_records: int
    duplicate_records_found: int
    annotated_class_counts: Dict[str, int]


def compute_waveform_hash(waveform: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash of a waveform array to detect exact duplicates."""
    contiguous = np.ascontiguousarray(np.round(waveform, decimals=5))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def audit_cohort_manifest(df: pd.DataFrame, cohort_name: str, provenance_group: str) -> CohortAuditSummary:
    """Summarize record counts, patient IDs, and label coverage for a candidate ECG cohort."""
    total_records = len(df)
    unique_patients = df["patient_id"].nunique() if "patient_id" in df.columns else None
    usable_12_lead = int(df["has_full_12_leads"].sum()) if "has_full_12_leads" in df.columns else total_records
    duplicates = int(df["waveform_hash"].duplicated().sum()) if "waveform_hash" in df.columns else 0

    return CohortAuditSummary(
        cohort_name=cohort_name,
        provenance_group=provenance_group,
        total_records=total_records,
        unique_patients=unique_patients,
        usable_12_lead_records=usable_12_lead,
        duplicate_records_found=duplicates,
        annotated_class_counts={},
    )
