"""
Stochastic Weight Averaging (SWA)

Averaging Weights Leads to Wider Optima and Better Generalization

https://github.com/timgaripov/swa
"""
from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader
from .utils import choose_device
from tqdm import tqdm


def moving_average(net1, net2, alpha=1.):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, device):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    model = model.to(device)
    pbar = tqdm(loader, unit="samples", unit_scale=loader.batch_size)
    for batch in pbar:
        images = batch.get('input', None)
        if images is None:
            warnings.warn("empty inputs")
            continue

        images = images.to(device)
        b = images.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(images)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def swa(load_model, model_folder, dataset, batch_size, device):
    directory = Path(model_folder)
    model_files = [f for f in directory.iterdir() if str(f).endswith(".model.pth")]
    assert(len(model_files) > 1)

    net = load_model(model_files[0])
    for i, f in enumerate(model_files[1:]):
        net2 = load_model(f)
        moving_average(net, net2, 1. / (i + 2))

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    device = choose_device(device)
    with torch.no_grad():
        bn_update(dataloader, net, device)
    return net