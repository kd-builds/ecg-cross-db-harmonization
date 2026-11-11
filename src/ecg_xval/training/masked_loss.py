"""
Masked binary cross-entropy loss that respects per-record, per-class eligibility masks.
"""
import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Computes BCEWithLogitsLoss while masking out classes that are unannotated or ineligible
    for a given record (eval_mask == 0), preventing silent conversion of missing diagnoses
    into confirmed negatives.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, eval_mask: torch.Tensor) -> torch.Tensor:
        raw_loss = self.bce(logits, targets)
        masked_loss = raw_loss * eval_mask
        denom = torch.clamp(eval_mask.sum(), min=1.0)
        return masked_loss.sum() / denom
