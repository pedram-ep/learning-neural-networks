"""Multi-task loss: weighted sum of CrossEntropy and BCEWithLogits."""

import torch
import torch.nn as nn
import config


class MultiTaskLoss(nn.Module):
    """
    L_total = ALPHA * CrossEntropy(logits_a, label_a)
            + BETA  * BCEWithLogits(logits_b, label_b)
    """

    def __init__(self, alpha: float = None, beta: float = None):
        super().__init__()
        self.alpha = alpha if alpha is not None else config.ALPHA
        self.beta = beta if beta is not None else config.BETA

        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits_a, logits_b, label_a, label_b):
        # logits_a: (B, 3), label_a: (B,)   -> CrossEntropyLoss expects this
        loss_a = self.ce(logits_a, label_a)

        # logits_b: (B, 1), label_b: (B,)   -> make both (B, 1)
        loss_b = self.bce(logits_b, label_b.view(-1, 1).float())

        total = self.alpha * loss_a + self.beta * loss_b
        return total, loss_a, loss_b
    
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=None, beta=None,
                 weight_a=None, pos_weight_b=None):
        super().__init__()
        self.alpha = alpha if alpha is not None else config.ALPHA
        self.beta  = beta  if beta  is not None else config.BETA
        self.ce  = nn.CrossEntropyLoss(weight=weight_a)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_b)

    def forward(self, logits_a, logits_b, label_a, label_b):
        loss_a = self.ce(logits_a, label_a)
        loss_b = self.bce(logits_b, label_b.view(-1, 1).float())
        return self.alpha * loss_a + self.beta * loss_b, loss_a, loss_b