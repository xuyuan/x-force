
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basenet import create_basenet, BASENET_CHOICES
from .attention import DANetHead
from .oc_net import BaseOC
from .inplace_abn import ActivatedBatchNorm
from .layer import ConvBnRelu, ConvBn, ConvRelu
from .scse import SCSEBlock
from data.coco import mask_to_class_segments


class UpsamplingBilinear(nn.Module):
    def forward(self, input):
        return F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)


class DecoderBase(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = self._init(in_channels, middle_channels, out_channels)

    def _init(self, in_channels, middle_channels, out_channels):
        raise NotImplementedError

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class DecoderDeConv(DecoderBase):
    def _init(self, in_channels, middle_channels, out_channels):
        return nn.Sequential(
            #nn.Dropout2d(p=0.1, inplace=True),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            ActivatedBatchNorm(middle_channels),
            DANetHead(middle_channels, middle_channels),
            # Parameters were chosen to avoid artifacts, suggested by https://distill.pub/2016/deconv-checkerboard/
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            #upsample(scale_factor=2)
        )


class DecoderSimple(DecoderBase):
    """as dsb2018_topcoders
    from https://github.com/selimsef/dsb2018_topcoders/blob/master/selim/models/unets.py#L68
    """
    def _init(self, in_channels, middle_channels, out_channels):
        return nn.Sequential(
            ConvBnRelu(in_channels, middle_channels, kernel_size=3, padding=1),
            UpsamplingBilinear(),
            ConvBn(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
            )


class DecoderSimpleNBN(DecoderBase):
    """as dsb2018_topcoders
    from https://github.com/selimsef/dsb2018_topcoders/blob/master/selim/models/unets.py#L76
    """
    def _init(self, in_channels, middle_channels, out_channels):
        return nn.Sequential(
            ConvRelu(in_channels, middle_channels, kernel_size=3, padding=1),
            UpsamplingBilinear(),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )


class DecoderSCSE(DecoderBase):
    """
    https://github.com/SeuTao/TGS-Salt-Identification-Challenge-2018-_4th_place_solution/blob/master/model/model.py#L125
    """
    def _init(self, in_channels, middle_channels, out_channels):
        return nn.Sequential(
            ConvBnRelu(in_channels, middle_channels, kernel_size=3, padding=1),
            ConvBnRelu(middle_channels, out_channels, kernel_size=3, padding=1),
            SCSEBlock(out_channels),
            UpsamplingBilinear()
        )


class ConcatPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.ap = nn.AvgPool2d(kernel_size, stride)
        self.mp = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return torch.cat((self.mp(x), self.ap(x)), 1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=(1,1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat((self.mp(x), self.ap(x)), 1)


class UNetBase(nn.Module):
    def __init__(self, basenet='vgg11', num_filters=16, pretrained='imagenet', upscale_input=True,
                 align_corners=False, Decoder=DecoderDeConv):
        super().__init__()
        self.align_corners = align_corners
        net, bn, n_pretrained = create_basenet(basenet, pretrained)

        if basenet.startswith('vgg') or not upscale_input:
            self.encoder1 = net[0]  # 64
        else:
            # add upsample
            self.encoder1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                net[0])
            self.encoder1.out_channels = net[0].out_channels

        self.encoder2 = net[1]  # 64
        self.encoder3 = net[2]  # 128
        self.encoder4 = net[3]  # 256

        context_channels = num_filters * 8 * 4
        self.encoder5 = nn.Sequential(
            net[4],
            nn.Conv2d(net[4].out_channels, context_channels, kernel_size=3, stride=1, padding=1),
            ActivatedBatchNorm(context_channels, activation='none'),
            BaseOC(in_channels=context_channels, out_channels=context_channels,
                   key_channels=context_channels // 2,
                   value_channels=context_channels // 2,
                   dropout=0.05)
        )
        self.encoder5.out_channels = context_channels

        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decoder(self.encoder5.out_channels, num_filters * 8 * 2, num_filters * 8)

        self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)

        if basenet.startswith('vgg'):
            self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)
            self.decoder1 = nn.Sequential(
                nn.Conv2d(self.encoder1.out_channels + num_filters, num_filters, kernel_size=3, padding=1),
                nn.ReLU(inplace=True))
        else:
            self.decoder2 = nn.Sequential(
                nn.Conv2d(self.encoder2.out_channels + num_filters * 2, num_filters * 2 * 2, kernel_size=3, padding=1),
                ActivatedBatchNorm(num_filters * 2 * 2),
                nn.Conv2d(num_filters * 2 * 2, num_filters, kernel_size=3, padding=1),
                ActivatedBatchNorm(num_filters))
            self.decoder1 = Decoder(self.encoder1.out_channels + num_filters, num_filters * 2, num_filters)

        self.decoder_channels = (1, 1, 2, 4, 8)

    def forward(self, x):
        # depth encoding / CoordConv
        # coord_scale = 1 / std[0]
        # coord_x = (torch.abs(torch.linspace(-1, 1, steps=x.size(3))) - 0.5) * coord_scale
        # x[:, 1] = coord_x.unsqueeze(0).expand_as(x[:, 1])
        # coord_y = (torch.linspace(-1, 1, steps=x.size(2))) * coord_scale
        # x[:, 2] = coord_y.unsqueeze(-1).expand_as(x[:, 2])

        e1 = self.encoder1(x)#; print('e1', e1.size())
        e2 = self.encoder2(e1)#; print('e2', e2.size())
        e3 = self.encoder3(e2)#; print('e3', e3.size())
        e4 = self.encoder4(e3)#; print('e4', e4.size())
        e5 = self.encoder5(e4)#; print('e5', e5.size())

        c = self.center(self.pool(e5))#; print('c', c.size())

        d5 = self.decoder5(c, e5)#; print('d5', d5.size())
        d4 = self.decoder4(d5, e4)#; print('d4', d4.size())
        d3 = self.decoder3(d4, e3)#; print('d3', d3.size())
        d2 = self.decoder2(torch.cat((d3, e2), 1))#; print('d2', d2.size())
        d1 = self.decoder1(d2, e1)#; print('d1', d1.size())

        d1_size = d1.size()[2:]
        upsampler = functools.partial(F.interpolate, size=d1_size, mode='bilinear',
                                      align_corners=self.align_corners)
        u5 = upsampler(d5)
        u4 = upsampler(d4)
        u3 = upsampler(d3)
        u2 = upsampler(d2)

        outputs = d1, u2, u3, u4, u5
        outputs = tuple(F.dropout2d(v, p=0.5, training=self.training, inplace=True) for v in outputs)

        return outputs


class UNet(UNetBase):
    def __init__(self, classnames, basenet='vgg11', num_filters=16, pretrained='imagenet',
                 upscale_input=True, align_corners=False, Decoder=DecoderDeConv,
                 num_logit_features=None, logit_bias=True, logit_bn=True):
        super().__init__(basenet=basenet, num_filters=num_filters, pretrained=pretrained,
                         upscale_input=upscale_input, align_corners=align_corners, Decoder=Decoder)
        self.num_filters = num_filters
        self.classnames = classnames
        self.logit_bias = logit_bias
        self.logit_bn = logit_bn
        num_classes = len(self.classnames)

        if num_logit_features is None:
            num_logit_features = num_filters * num_classes
        self.num_logit_features = num_logit_features

        self._init()

        #self.global_gate = nn.Sequential(AdaptiveConcatPool2d(),
        #                                 nn.Conv2d(self.encoder5.out_channels * 2, 1, kernel_size=1),
        #                                 nn.Sigmoid()
        #                                 )
        #self.global_logit = nn.Sequential(nn.Conv2d(num_filters + 1, 1, kernel_size=1),
        #                                  )

    def create_logit_module(self, n):
        num_classes = len(self.classnames)
        modules = [nn.Dropout2d(p=0, inplace=True),
                   nn.Conv2d(self.num_filters * n, self.num_logit_features, kernel_size=3, padding=1),
                    # ----------------------------------------------------------------------------------
                    # super resolution
                    #nn.Conv2d(num_filters * (8 + 4 + 2 + 1 + 1), num_filters * 8, kernel_size=3, padding=1),
                    #nn.ReLU(inplace=True),
                    #nn.Conv2d(num_filters * 8, num_filters * 4, kernel_size=3, padding=1),
                    #nn.ReLU(inplace=True),
                    #Decoder(num_filters * 4, num_filters * 2, num_filters),
                    #ConcatPool2d(2, 2),
                    #nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, padding=1),
                    # ----------------------------------------------------------------------------------
                    ]
        if self.logit_bn:
            modules += [ActivatedBatchNorm(self.num_logit_features)]
        else:
            modules += [nn.ReLU(inplace=True)]
        modules += [nn.Conv2d(self.num_logit_features, num_classes, kernel_size=1, bias=self.logit_bias),
                    #nn.Sigmoid()
                    ]
        return  nn.Sequential(*modules)

    def _init(self):
        self.logit = self.create_logit_module(sum(self.decoder_channels))

    def forward(self, x):
        d1, u2, u3, u4, u5 = super().forward(x)
        d = torch.cat((d1, u2, u3, u4, u5), 1)
        logit = self.logit(d)#;print(logit.size())

        #g = self.global_gate(e5)
        #g = nn.Upsample(size=d1_size, mode='nearest')(g)
        #logit = self.global_logit(torch.cat((logit, g), 1))

        logit = F.interpolate(logit, size=x.size()[2:],
                              mode='bilinear', align_corners=self.align_corners)
        return logit

    def mask_criterion(self, outputs, targets, loss_func=F.binary_cross_entropy_with_logits,
                       upsample=True):
        if upsample:
            # upsample outputs
            image_size = targets.shape[-2:]
            outputs = F.interpolate(outputs, size=image_size)
        else:
            # downsample targets
            outputs_size = outputs.shape[-2:]
            targets = F.interpolate(targets, size=outputs_size)

        outputs = outputs.float()

        c = outputs.size(1)
        losses = {}
        for i, t in enumerate(mask_to_class_segments(targets, c)):
            o = outputs[:, i]
            loss = loss_func(o, t.float())
            losses[self.classnames[i]] = loss
        return losses

    def criterion(self, outputs, targets, loss_func=F.binary_cross_entropy_with_logits, upsample=True):
        return self.mask_criterion(outputs, targets['masks'], loss_func, upsample)



class UNetDS(UNet):
    def _init(self):
        super()._init()
        self.logit_ds = nn.ModuleList([self.create_logit_module(n) for n in self.decoder_channels])

    def forward(self, x):
        y = UNetBase.forward(self, x)
        d = torch.cat(y, 1)
        logit = self.logit(d)#;print(logit.size())

        if self.training:
            logit_ds = [m(yi) for m, yi in zip(self.logit_ds, y)]
            return logit, logit_ds
        else:
            return logit

    def train(self, mode=True):
        return super().train(mode)

    def criterion(self, outputs, targets, loss_func=F.binary_cross_entropy_with_logits, upsample=True):
        if not self.training or torch.is_tensor(outputs):
            return super().criterion(outputs, targets, loss_func, upsample)

        logit, logit_ds = outputs
        losses = super().criterion(logit, targets, loss_func, upsample)
        s = 2 ** -len(logit_ds) * 0.5
        for lds in logit_ds:
            losses_lds = super().criterion(lds, targets, loss_func, upsample)
            for i, k in enumerate(losses):
                losses[k] = losses[k] + losses_lds[k] * (s * (2 ** i))
        return losses


class UNetEdge(UNet):
    def _init(self):
        num_classes = len(self.classnames)

        modules = [nn.Dropout2d(p=0, inplace=True),
                   nn.Conv2d(self.num_filters * sum(self.decoder_channels), self.num_logit_features, kernel_size=3, padding=1)
                   ]
        if self.logit_bn:
            modules += [ActivatedBatchNorm(self.num_logit_features)]
        else:
            modules += [nn.ReLU(inplace=True)]

        self.fuse = nn.Sequential(*modules)
        self.logit = nn.Conv2d(self.num_logit_features, num_classes, kernel_size=1, bias=self.logit_bias)
        self.logit_edge = nn.Conv2d(self.num_logit_features, num_classes, kernel_size=1, bias=self.logit_bias)

    def forward(self, x):
        y = UNetBase.forward(self, x)
        d = torch.cat(y, 1)
        z = self.fuse(d)
        logit = self.logit(z)

        if self.training:
            logit_e = self.logit_edge(z)
            return logit, logit_e
        else:
            return logit

    def train(self, mode=True):
        return super().train(mode)

    def criterion(self, outputs, targets, loss_func=F.binary_cross_entropy_with_logits, upsample=True):
        if torch.is_tensor(outputs):
            return self.mask_criterion(outputs, targets['masks'], loss_func, upsample)

        logit, logit_e = outputs
        losses = self.mask_criterion(logit, targets['masks'], loss_func, upsample)
        losses_e = self.mask_criterion(logit_e, targets['edges'], loss_func, upsample)
        for k in losses:
            losses[k] = losses[k] + losses_e[k]
        return losses


class UNetEdgeW(UNetEdge):
    def criterion(self, outputs, targets, loss_func=F.binary_cross_entropy_with_logits, upsample=True):
        if torch.is_tensor(outputs):
            return self.mask_criterion(outputs, targets['masks'], loss_func, upsample)

        logit, logit_e = outputs
        losses = self.mask_criterion_with_weight(logit, targets['masks'], targets['edges'], loss_func, upsample)
        losses_e = self.mask_criterion(logit_e, targets['edges'], loss_func, upsample)
        for k in losses:
            losses[k] = losses[k] + losses_e[k]
        return losses

    def mask_criterion_with_weight(self, outputs, targets, weights, loss_func=F.binary_cross_entropy_with_logits, upsample=True):
        if upsample:
            # upsample outputs
            image_size = targets.shape[-2:]
            outputs = F.interpolate(outputs, size=image_size)
        else:
            # downsample targets
            outputs_size = outputs.shape[-2:]
            targets = F.interpolate(targets, size=outputs_size)

        outputs = outputs.float()

        c = outputs.size(1)
        losses = {}
        for i, (t, w) in enumerate(zip(mask_to_class_segments(targets, c),
                                       mask_to_class_segments(weights, c))):
            o = outputs[:, i]
            w = 1 + w.float()
            loss = loss_func(o, t.float(), weight=w)
            losses[self.classnames[i]] = loss
        return losses


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--basenet", choices=BASENET_CHOICES, default='vgg11', help='model of basenet')
    parser.add_argument("--num-filters", type=int, default=16, help='num filters for decoder')
    parser.add_argument("--num_classes", type=int, default=5, help='num of output classes')
    parser.add_argument("--upscale-input", action='store_true',
                        help='scale input to make output the same size as original input')

    args = parser.parse_args()

    net = UNet(**vars(args))
    #print(net)
    parameters = [p for p in net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in parameters)
    print('N of parameters {} ({} tensors)'.format(n_params, len(parameters)))
    encoder_parameters = [p for name, p in net.named_parameters() if p.requires_grad and name.startswith('encoder')]
    n_encoder_params = sum(p.numel() for p in encoder_parameters)
    print('N of encoder parameters {} ({} tensors)'.format(n_encoder_params, len(encoder_parameters)))
    print('N of decoder parameters {} ({} tensors)'.format(n_params - n_encoder_params, len(parameters) - len(encoder_parameters)))

    x = torch.empty((1, 3, 128, 128))
    y = net(x)
    print(x.size(), '-->', y.size())