"""
Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz: mixup: Beyond Empirical Risk Minimization
https://arxiv.org/abs/1710.09412
"""

import numpy as np
import torch


def index_targets(targets, index):
    if isinstance(targets, torch.Tensor):
        targets = targets[index]
    elif isinstance(targets, list):
        targets = [targets[i] for i in index]
    elif isinstance(targets, dict):
        targets = {k: index_targets(v, index) for k, v in targets.items()}
    else:
        raise NotImplementedError(type(targets))
    return targets


def mixup(inputs, alpha, criterion):
    s = inputs.size()[0]
    weight = np.random.beta(alpha, alpha)
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index]
    inputs = weight * x1 + (1 - weight) * x2

    def mixup_criterion(outputs, targets):
        losses1 = criterion(outputs, targets)
        targets2 = index_targets(targets, index)
        losses2 = criterion(outputs, targets2)
        losses = {k: weight * losses1[k] + (1 - weight) * losses2[k] for k in losses1}
        return losses

    return inputs, mixup_criterion

