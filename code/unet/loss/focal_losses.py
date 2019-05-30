"""focal loss"""

import torch
import torch.nn.functional as F


def binary_focal_loss_with_logits(input, target, focusing=2, weight=None, size_average=True, reduce=True):
    """
    focusing (float): parameter of Focal Loss (0 equals standard cross entropy)
    other parameters: see http://pytorch.org/docs/master/nn.html#torch.nn.functional.cross_entropy
    """
    p = torch.sigmoid(input)
    l = F.binary_cross_entropy(p, target, weight=weight, reduce=False)
    #neg_p = p * (1 - target) + (1 - p) * target
    neg_p = p - p * target * 2 + target

    fl = l * (neg_p ** focusing)
    if not reduce:
        return fl
    if not size_average:
        return torch.sum(fl)
    return fl.mean()
