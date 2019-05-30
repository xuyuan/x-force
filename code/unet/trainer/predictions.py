import shelve
import pickle
from pathlib import Path


class Predictions(object):
    def __init__(self, predictions, classnames=None, mode='r'):
        """
        :param predictions: list of `dict` or `str`
        :param classnames: list of `str`
        """
        self.predictions = predictions
        self.classnames = classnames
        self.datasets = []
        for p in self.predictions:
            if isinstance(p, dict):
                self.datasets.append(p)
            elif isinstance(p, str):
                flag = 'r' if mode == 'r' else 'c'
                self.datasets.append(shelve.open(p, flag=flag))
            else:
                raise RuntimeError('unknown type {0}'.format(type(p)))
        self.mode = mode

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, key):
        for det in self.datasets:
            if key in det:
                return det[key]
        return None

    def __setitem__(self, key, value):
        assert self.mode == 'w'
        self.datasets[0][key] = value

    def __iter__(self):
        for det in self.datasets:
            for image_id in det:
                yield image_id, det[image_id]

    def save(self, filename):
        state_dict = dict(predictions=self.predictions, classnames=self.classnames)
        pickle.dump(state_dict, open(filename, 'wb'))

    @staticmethod
    def open(filename, mode='r', n_new_file=0, classnames=None):
        if mode == 'r':
            state_dict = pickle.load(open(filename, 'rb'))
            # fix relative path
            state_dict['predictions'] = [str(Path(filename).parent / Path(f).name) if isinstance(f, str) else f
                                             for f in state_dict['predictions']]

            return Predictions(**state_dict)
        elif mode == 'w':
            assert classnames is not None
            dataset_file = str(Path(filename).parent / Path(filename).stem) + '{}'
            predictions = [dataset_file.format(i) for i in range(n_new_file)]
            if not predictions:
                predictions.append(dict())

            p = Predictions(predictions, classnames, mode='w')
            p.save(filename)
            return p
        else:
            raise RuntimeError("unknown mode {0}".format(mode))
