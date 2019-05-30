import torchvision.transforms as tvt


def _compose_repr(self, args_string=''):
    format_string = self.__class__.__name__ + '('
    indent = ' ' * len(format_string)
    trans_strings = [t.__repr__().replace('\n', '\n' + indent) for t in self.transforms]
    if args_string:
        trans_strings.insert(0, args_string)

    format_string += (',\n'+indent).join(trans_strings)
    format_string += ')'
    return format_string


class Compose(tvt.Compose):
    def __repr__(self): return _compose_repr(self)


class RandomApply(tvt.RandomApply):
    def __init__(self, transforms, p=0.5):
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]
        super().__init__(transforms, p=p)

    def __repr__(self):
        format_string = 'p={}'.format(self.p)
        return _compose_repr(self, format_string)


class RandomOrder(tvt.RandomOrder):
    def __repr__(self): return _compose_repr(self)


class RandomChoice(tvt.RandomChoice):
    def __repr__(self): return _compose_repr(self)


class PassThough(object):
    def __call__(self, sample): return sample

    def __repr__(self): return self.__class__.__name__ + '()'


class FilterSample(object):
    def __init__(self, keep_keys):
        self.keep_keys = keep_keys

    def __call__(self, sample):
        return {k: v for k, v in sample.items() if k in self.keep_keys}

    def __repr__(self):
        return self.__class__.__name__ + '(' + repr(self.keep_keys) + ')'


class ApplyOnly(object):
    def __init__(self, keys, transform):
        self.keys = keys
        self.transform = transform

    def __call__(self, sample):
        filtered_sample = {k: v for k, v in sample.items() if k in self.keys}
        excluded_sample = {k: v for k, v in sample.items() if k not in self.keys}
        sample = self.transform(filtered_sample)
        sample.update(excluded_sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(keys=' + repr(self.keys) + ', transform=' + repr(self.transform) + ')'
