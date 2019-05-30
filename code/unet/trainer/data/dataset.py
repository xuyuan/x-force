
from random import randint
import torch
import torch.utils.data as tud


class AttributeMissingMixin(object):
    """ A Mixin' to implement the 'method_missing' Ruby-like protocol. """
    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError as e:
            if attr.startswith('__'):
                raise e

            return self._attribute_missing(attr=attr)

    def _attribute_missing(self, attr):
        """ This method should be overridden in the derived class. """
        raise NotImplementedError(self.__class__.__name__ + " '_attribute_missing' method has not been implemented.")


class Dataset(tud.Dataset):
    def __add__(self, other):
        return ConcatDataset([self, other])


class ConcatDataset(Dataset, tud.ConcatDataset, AttributeMissingMixin):
    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        indent = ' ' * 4
        fmt_str += (indent + 'len: {}\n'.format(len(self)))
        for dataset in self.datasets:
            dataset_str_lines = str(dataset).split('\n')
            dataset_str_lines = ['-' * len(dataset_str_lines[0])] + dataset_str_lines
            dataset_str_lines = [indent + s for s in dataset_str_lines if s]
            fmt_str += '\n'.join(dataset_str_lines)
            fmt_str += '\n'

        return fmt_str

    def _attribute_missing(self, attr):
        """forward missing attr to dataset[0]"""
        return getattr(self.datasets[0], attr)


class Subset(Dataset, tud.Subset, AttributeMissingMixin):
    """
    https://github.com/pytorch/vision/issues/369
    """
    def __init__(self, dataset, indices):
        if isinstance(indices, slice):
            indices = range(len(dataset))[indices]

        super().__init__(dataset, indices)

    def _attribute_missing(self, attr):
        """forward missing attr to dataset"""
        return getattr(self.dataset, attr)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        indent = ' ' * 4
        fmt_str += (indent + 'len: {}\n'.format(len(self)))

        dataset_str_lines = str(self.dataset).split('\n')
        dataset_str_lines = ['-' * len(dataset_str_lines[0])] + dataset_str_lines
        dataset_str_lines = [indent + s for s in dataset_str_lines if s]
        fmt_str += '\n'.join(dataset_str_lines)
        fmt_str += '\n'

        return fmt_str


class RollSplitSet(Subset):
    def __init__(self, dataset, n_split):
        self.n_split = n_split
        self.i_split = 0
        self.end_split = len(dataset) - (len(dataset) % self.n_split)
        self.get_count = 0
        indices = slice(self.i_split, self.end_split, self.n_split)
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        # use get count as indicator for rolling to next split
        self.get_count += 1
        if self.get_count == len(self):
            self.i_split += 1
            self.i_split %= self.n_split
            print("RollSplitSet roll to next {}".format(self.i_split))
            indices = slice(self.i_split, self.end_split, self.n_split)
            self.indices = range(len(self.dataset))[indices]
            self.get_count = 0

        return item


class RandomSubset(Subset):
    def __init__(self, dataset, size):
        super().__init__(dataset, None)
        self.size = size

    def __getitem__(self, idx):
        idx = randint(0, len(self.dataset) - 1)
        return self.dataset[idx]

    def __len__(self):
        return self.size


class ShuffledDataset(Subset):
    def __init__(self, dataset):
        indices = torch.randperm(len(dataset))
        super().__init__(dataset, indices)


class TransformedDataset(Dataset, AttributeMissingMixin):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.transform(sample)

    def __len__(self):
        return len(self.dataset)

    def _attribute_missing(self, attr):
        """forward missing method to dataset"""
        return getattr(self.dataset, attr)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        indent = ' ' * 4
        dataset_lines = repr(self.dataset).split('\n')
        dataset_lines = [indent + s for s in dataset_lines if s]
        fmt_str += '\n'.join(dataset_lines)
        tmp = '\n    Transforms: '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__str__().replace('\n', '\n' + ' ' * (len(tmp)-1)))
        return fmt_str
