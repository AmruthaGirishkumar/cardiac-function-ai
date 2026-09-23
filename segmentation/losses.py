"""
losses.py
---------
Loss functions for LV segmentation training.

Per Section 6.1 of the technical spec: we combine Binary Cross-Entropy (BCE)
with Dice loss. BCE alone treats every pixel equally, but the left ventricle
is a small region against a large background, so a model can get deceptively
low BCE just by predicting "background" everywhere. Dice loss is computed on
the overlap between prediction and ground truth, so it directly rewards
getting the (small) foreground region right and pulls training back toward
that class balance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Dice = 2 * |pred ∩ target| / (|pred| + |target|)
    Loss = 1 - Dice   (so 0 = perfect overlap, 1 = no overlap)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and Dice loss.

    bce_weight / dice_weight let you tilt training toward one term if needed
    (e.g. more Dice weight if the mask keeps collapsing to empty).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


@torch.no_grad()
def dice_coefficient(pred_mask: torch.Tensor, target_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    """Dice score between two already-thresholded binary masks (0/1). Used at eval time."""
    pred = pred_mask.view(-1).float()
    target = target_mask.view(-1).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return ((2.0 * intersection + smooth) / (union + smooth)).item()


@torch.no_grad()
def iou_score(pred_mask: torch.Tensor, target_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    """Intersection-over-Union between two already-thresholded binary masks (0/1)."""
    pred = pred_mask.view(-1).float()
    target = target_mask.view(-1).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()
