
import functools

import torch
from torch import nn
from .unet import UNet, BASENET_CHOICES, UNetDS, UNetEdge, UNetEdgeW, DecoderSimple, DecoderSCSE, DecoderSimpleNBN
from .attention import DANet

xray_classnames = ('铁壳打火机', '黑钉打火机', '刀具', '电源和电池', '剪刀')


def create(model, basenet, classnames=xray_classnames, pretrained='imagenet'):
    print(f'nn.create(model={model}, basenet={basenet})')
    num_classes = len(classnames)
    if model == 'unet':
        net = UNet(classnames, basenet=basenet, pretrained=pretrained)
    elif model == 'unet2':
        net = UNet(classnames, basenet=basenet, pretrained=pretrained,
                   num_logit_features=num_classes*32,
                   logit_bias=False)
    elif model == 'unet2_e':
        net = UNetEdge(classnames, basenet=basenet, pretrained=pretrained,
                       num_logit_features=num_classes*32,
                       logit_bias=False)
    elif model == 'unet2_e_simple':
        net = UNetEdge(classnames, basenet=basenet, pretrained=pretrained,
                       Decoder=DecoderSimple,
                       num_logit_features=num_classes*32,
                       logit_bias=False)
    elif model == 'unet2_e_simple_nbn':
        net = UNetEdge(classnames, basenet=basenet, pretrained=pretrained,
                       Decoder=DecoderSimpleNBN,
                       num_logit_features=num_classes*32,
                       logit_bias=False, logit_bn=False)
    elif model == 'unet2_e_scse':
        net = UNetEdge(classnames, basenet=basenet, pretrained=pretrained,
                       Decoder=DecoderSCSE,
                       num_logit_features=num_classes*32,
                       logit_bias=False)
    elif model == 'unet2_ew_scse':
        net = UNetEdgeW(classnames, basenet=basenet, pretrained=pretrained,
                       Decoder=DecoderSCSE,
                       num_logit_features=num_classes*32,
                       logit_bias=False)
    #######################################################################
    elif model == 'unet2_ds':
        net = UNetDS(classnames, basenet=basenet, pretrained=pretrained,
                     num_logit_features=num_classes*32,
                     logit_bias=False)
    elif model == 'unet3':
        net = UNet(classnames, basenet=basenet, pretrained=pretrained,
                   num_logit_features=num_classes*32,
                   logit_bias=False, upscale_input=False)
    elif model == 'unet3_ds':
        net = UNetDS(classnames, basenet=basenet, pretrained=pretrained,
                     num_logit_features=num_classes*32,
                     logit_bias=False, upscale_input=False)
    elif model == 'danet':
        net = DANet(basenet=basenet, pretrained=pretrained)
    else:
        raise NotImplementedError(model)

    net.args = dict(model=model, basenet=basenet, pretrained=pretrained)
    net.save = functools.partial(save, net)
    net.load = load

    return net


def save(net, filename):
    if isinstance(net, nn.DataParallel):
        net = net.module

    data = dict(args=net.args,
                state_dict=net.state_dict())
    torch.save(data, filename)


def load(filename):
    print('load {}'.format(filename))
    data = torch.load(filename, map_location='cpu')
    data['args']['pretrained'] = filename
    net = create(**data['args'])
    net.load_state_dict(data['state_dict'])
    return net
