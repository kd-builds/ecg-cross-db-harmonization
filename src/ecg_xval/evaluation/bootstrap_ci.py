"""
Patient-cluster bootstrap 95% confidence intervals for external evaluation.
Corresponds to Section 7 of the research proposal.
"""
from typing import Callable, Tuple
import numpy as np


def patient_cluster_bootstrap_ci(
    patient_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Resamples unique patient clusters with replacement to compute a 95% confidence interval.
    """
    rng = np.random.default_rng(random_seed)
    unique_patients = np.unique(patient_ids)
    patient_to_indices = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}

    estimates = []
    for _ in range(n_resamples):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled_indices = np.concatenate([patient_to_indices[pid] for pid in sampled_patients])
        val = metric_fn(sampled_indices)
        if not np.isnan(val):
            estimates.append(val)

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(estimates, 100 * alpha))
    upper = float(np.percentile(estimates, 100 * (1.0 - alpha)))
    return lower, upper
