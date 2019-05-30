from .dataset import *
from .image_folder import ImageFolder, get_image_files


def split_dataset(dataset, num_processes, balanced=False):
    if balanced:
        dataset_splits = [Subset(dataset, slice(start, None, num_processes)) for start in range(num_processes)]
    else:
        # split in the way of keeping order
        dataset_splits = []
        splited_size = len(dataset) // num_processes
        for i in range(num_processes):
            start = i * splited_size
            end = start + splited_size
            dataset_splits.append([start, end])
        dataset_splits[-1][-1] = len(dataset)
        dataset_splits = [Subset(dataset, slice(start, end)) for start, end in dataset_splits]
    return dataset_splits
