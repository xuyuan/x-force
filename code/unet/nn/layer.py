import torch.nn as nn


def conv(*args, **kwargs):
    return lambda last_layer: nn.Conv2d(get_num_of_channels(last_layer), *args, **kwargs)


def get_num_of_channels(layers, channle_name='out_channels'):
    """access out_channels from last layer of nn.Sequential/list"""
    if hasattr(layers, channle_name):
        return getattr(layers, channle_name)
    elif isinstance(layers, int):
        return layers
    else:
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            if hasattr(layer, 'out_channels'):
                return getattr(layer, channle_name)
            elif isinstance(layer, nn.Sequential):
                return get_num_of_channels(layer, channle_name)
    raise RuntimeError("cant get_num_of_channels {} from {}".format(channle_name, layers))


def Sequential(*args):
    f = nn.Sequential(*args)
    f.in_channels = get_num_of_channels(args, 'in_channels')
    f.out_channels = get_num_of_channels(args)
    return f


def sequential(*args):
    def create_sequential(last_layer):
        layers = []
        for a in args:
            layers.append(a(last_layer))
            last_layer = layers[-1]
        return Sequential(*layers)
    return create_sequential


def ConvBn(*args, **kwargs):
    """drop in block for nn.Conv2d with BatchNorm and ReLU"""
    c = nn.Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels))


def conv_bn(*args, **kwargs):
    return lambda last_layer: ConvBn(get_num_of_channels(last_layer), *args, **kwargs)


def ConvBnRelu(*args, **kwargs):
    """drop in block for nn.Conv2d with BatchNorm and ReLU"""
    c = nn.Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels),
                      nn.ReLU(inplace=True))


def conv_bn_relu(*args, **kwargs):
    return lambda last_layer: ConvBnRelu(get_num_of_channels(last_layer), *args, **kwargs)


def ConvBnRelu6(*args, **kwargs):
    """drop in block for nn.Conv2d with BatchNorm and ReLU6"""
    c = nn.Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels),
                      nn.ReLU(inplace=True))


def conv_bn_relu6(*args, **kwargs):
    return lambda last_layer: ConvBnRelu6(get_num_of_channels(last_layer), *args, **kwargs)


def ConvRelu(*args, **kwargs):
    return Sequential(nn.Conv2d(*args, **kwargs),
                      nn.ReLU(inplace=True))


def conv_relu(*args, **kwargs):
    return lambda last_layer: ConvRelu(get_num_of_channels(last_layer), *args, **kwargs)


def ConvRelu6(*args, **kwargs):
    return Sequential(nn.Conv2d(*args, **kwargs),
                      nn.ReLU6(inplace=True))


def conv_relu6(*args, **kwargs):
    return lambda last_layer: ConvRelu6(get_num_of_channels(last_layer), *args, **kwargs)


def ReluConv(*args, **kwargs):
    return Sequential(nn.ReLU(inplace=True),
                      nn.Conv2d(*args, **kwargs))


def relu_conv(*args, **kwargs):
    return lambda last_layer: ReluConv(get_num_of_channels(last_layer), *args, **kwargs)


def BnReluConv(*args, **kwargs):
    """drop in block for nn.Conv2d with BatchNorm and ReLU"""
    c = nn.Conv2d(*args, **kwargs)
    return Sequential(nn.BatchNorm2d(c.in_channels),
                      nn.ReLU(inplace=True),
                      c)


def bn_relu_conv(*args, **kwargs):
    return lambda last_layer: BnReluConv(get_num_of_channels(last_layer), *args, **kwargs)
