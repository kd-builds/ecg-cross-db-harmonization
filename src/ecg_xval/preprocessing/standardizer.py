"""
ECG waveform standardization: lead ordering, resampling, windowing, and train-only normalization.
Corresponds to Section 6 (Data standardization) of the research proposal.
"""
from typing import List, Optional
import numpy as np
from scipy.signal import resample


CANONICAL_12_LEADS: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
]


class ECGStandardizer:
    """
    Standardizes lead order, sampling rate, and fixed window length.
    Normalization statistics are fitted strictly on source training data to prevent leakage.
    """

    def __init__(self, target_fs: int = 500, window_seconds: float = 10.0):
        self.target_fs = target_fs
        self.target_length = int(target_fs * window_seconds)
        self.train_mean_: Optional[np.ndarray] = None
        self.train_std_: Optional[np.ndarray] = None

    def reorder_leads(self, signal: np.ndarray, source_leads: List[str]) -> np.ndarray:
        """Reorder input (C, L) signal to match CANONICAL_12_LEADS."""
        idx_map = [source_leads.index(lead) for lead in CANONICAL_12_LEADS]
        return signal[idx_map, :]

    def resample_and_window(self, signal: np.ndarray, source_fs: int) -> np.ndarray:
        """Resample (C, L) waveform to target_fs and pad/crop to target_length."""
        num_leads, num_samples = signal.shape
        if source_fs != self.target_fs:
            new_samples = int(round(num_samples * self.target_fs / source_fs))
            signal = resample(signal, new_samples, axis=1)

        cur_len = signal.shape[1]
        if cur_len >= self.target_length:
            return signal[:, : self.target_length].astype(np.float32)

        padded = np.zeros((num_leads, self.target_length), dtype=np.float32)
        padded[:, :cur_len] = signal
        return padded

    def fit_normalizer(self, train_signals: np.ndarray) -> "ECGStandardizer":
        """Fit channel-wise mean and std strictly on training waveforms (N, C, L)."""
        self.train_mean_ = train_signals.mean(axis=(0, 2), keepdims=True)
        self.train_std_ = train_signals.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Apply frozen training normalization statistics without target-data fitting."""
        if self.train_mean_ is None or self.train_std_ is None:
            raise RuntimeError("Normalizer must be fitted on source training data before transform.")
        return (signals - self.train_mean_) / self.train_std_
