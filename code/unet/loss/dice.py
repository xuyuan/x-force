import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def dice_loss(y_pred, y_true, smooth=1, eps=1e-7):
    product = y_pred * y_true
    intersection = product.sum()
    coefficient = (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth + eps)
    loss = 1. - coefficient
    # or "-coefficient"
    return loss


def dice_loss_with_logits(y_pred, y_true, smooth=1, eps=1e-7):
    y_pred = torch.sigmoid(y_pred)
    return dice_loss(y_pred, y_true, smooth, eps)


def dice_and_bce(y_pred, y_true, smooth=1, eps=1e-7):
    return dice_loss_with_logits(y_pred, y_true, smooth, eps) + binary_cross_entropy_with_logits(y_pred, y_true)