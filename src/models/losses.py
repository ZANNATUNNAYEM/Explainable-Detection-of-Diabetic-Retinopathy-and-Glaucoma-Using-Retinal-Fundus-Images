"""
Loss functions tailored for class-imbalanced retinal classification tasks.
Includes Focal Loss and Class-Weighted Cross Entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to counter extreme majority class bias (No DR).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # torch.Tensor of shape [C]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        target_p = p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow(1.0 - target_p, self.gamma)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha.gather(dim=0, index=targets)
            loss = -alpha_weight * focal_weight * target_log_p
        else:
            loss = -focal_weight * target_log_p

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(loss_type="focal", class_weights=None, gamma=2.0):
    """Factory method for loss functions."""
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    elif loss_type == "weighted_ce" and class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()
