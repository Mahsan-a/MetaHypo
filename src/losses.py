"""
losses.py
Focal loss for imbalanced binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    FL(p, y) = -α(1-p)^γ · y log p  -  (1-α) p^γ · (1-y) log(1-p)

    Down-weights easy normoglycemic days so gradient mass concentrates
    on misclassified hypoglycemic examples.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = logits.squeeze(-1)
        p       = torch.sigmoid(logits)
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t     = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()
