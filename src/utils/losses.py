import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        targets_one_hot = F.one_hot(targets, n_classes).float()

        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / n_classes

        loss = -(targets_smooth * log_probs).sum(dim=-1)

        if self.weight is not None:
            loss = loss * self.weight[targets]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(
    loss_type: str,
    class_weights: torch.Tensor = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    device: str = "mps"
):
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if loss_type == "ce_weighted":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == "ce_sqrt_weighted":
        if class_weights is not None:
            sqrt_weights = torch.sqrt(class_weights)
            sqrt_weights = sqrt_weights / sqrt_weights.sum() * len(sqrt_weights)
        else:
            sqrt_weights = None
        return nn.CrossEntropyLoss(weight=sqrt_weights)

    elif loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)

    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(epsilon=label_smoothing, weight=class_weights)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_class_weights(label_counts, method="inverse"):
    counts = np.array(list(label_counts.values()))

    if method == "inverse":
        weights = 1.0 / counts
    elif method == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    weights = weights / weights.sum() * len(weights)

    return torch.FloatTensor(weights)
