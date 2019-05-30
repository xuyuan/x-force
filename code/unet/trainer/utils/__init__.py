
import os
import torch

from .rle import rle_decode, rle_encoding

def choose_device(device):
    if not isinstance(device, str):
        return device

    if device not in ['cuda', 'cpu']:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        assert torch.cuda.is_available()

    device = torch.device(device)
    return device


def get_num_workers(jobs):
    """
    Parameters
    ----------
    jobs How many jobs to be paralleled. Negative or 0 means number of cpu cores left.

    Returns
    -------
    How many subprocess to be used
    """
    num_workers = jobs
    if num_workers <= 0:
        num_workers = os.cpu_count() + jobs
    if num_workers < 0 or num_workers > os.cpu_count():
        raise RuntimeError("System doesn't have so many cpu cores: {} vs {}".format(jobs, os.cpu_count()))
    return num_workers
