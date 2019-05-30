"""
test script
"""

import argparse
import functools
import warnings
import numpy as np
import torch
from torch.nn.functional import interpolate
from nn import load as load_model
from trainer import Tester


class FiveTile(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def crop(data, rect, add=None):
        if add is not None:
            data[..., rect[1]:rect[3], rect[0]:rect[2]] += add
        return data[..., rect[1]:rect[3], rect[0]:rect[2]]

    def apply(self, func, data):
        h, w = data.shape[2:4]
        sw, sh = self.size

        if h <= sh and w <= sw:
            return func(data)

        if h > sh * 2 or w > sw * 2:
            raise RuntimeError(f'input is too big {h, w}')

        if sw > w:
            sw = w
        if sh > h:
            sh = h

        tl = (0, 0, sw, sh)
        tr = (w - sw, 0, w, sh)
        bl = (0, h - sh, sw, h)
        br = (w - sw, h - sh, w, h)
        i = int(round((w - sw) / 2.))
        j = int(round((h - sh) / 2.))
        c  = (i, j, i + sw, j + sh)

        center = self.crop(data, c)
        center = func(center)
        nc = center.shape[1]
        output = center.new_zeros((1, nc, h, w))
        weights = center.new_zeros((1, 1, h, w))
        self.crop(output, c, add=center)
        self.crop(weights, c, add=1)

        rects = []
        if sw < w:
            rects += [tl, tr]
            if sh < h:
                rects += [bl, br]
        else:
            rects += [tl, bl]

        for rect in rects:
            input_rect = self.crop(data, rect)
            output_rect = func(input_rect)
            self.crop(output, rect, add=output_rect)
            self.crop(weights, rect, add=1)

        return output / weights


class NineTile(object):
    def __init__(self, size):
        self.size = size

    def crop(self, data, rect):
        return data[..., rect[1]:rect[1]+self.size[1], rect[0]:rect[0]+self.size[0]]

    def apply(self, func, data):
        h, w = data.shape[2:4]
        sw, sh = self.size

        if h <= sh and w <= sw:
            return func(data)

        if h > sh * 3 or w > sw * 3:
            raise RuntimeError(f'input is too big {h, w}')

        if sw > w:
            sw = w
        if sh > h:
            sh = h

        tl = (0, 0)
        tr = (w - sw, 0)
        bl = (0, h - sh)
        br = (w - sw, h - sh)
        i = int(round((w - sw) / 2.))
        j = int(round((h - sh) / 2.))
        c  = (i, j)
        tc = (i, 0)
        bc = (i, h - sh)
        lc = (0, j)
        rc = (w - sw, j)

        center = self.crop(data, c)
        center = func(center)
        nc = center.shape[1]
        output = center.new_zeros((1, nc, h, w)).cpu().float()
        weights = center.new_zeros((1, 1, h, w)).cpu().float()
        output_c = self.crop(output, c)
        output_c += center.cpu().float()
        weights_c = self.crop(weights, c)
        weights_c += 1

        rects = []
        if sw < w:
            rects += [tl, tc, tr]
            if sh < h:
                rects += [lc, rc, bl, bc, br]
        else:
            rects += [tl, lc, bl]

        for rect in rects:
            input_rect = self.crop(data, rect)
            output_rect = func(input_rect)
            output_c = self.crop(output, rect)
            weights_c = self.crop(weights, rect)
            output_c += output_rect.cpu().float()
            weights_c += 1

        return output / weights


class ArgumentParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--device', default='auto', choices=['gpu', 'cpu'], help='running with cpu or gpu')
        self.parser.add_argument('--half', action='store_true', help='running half float')

        self.parser.add_argument("-m", "--model", type=str, required=True, help='pre/trained SSD model file')
        self.parser.add_argument("-o", "--output", type=str, help='root directory of output')
        self.parser.add_argument("-i", "--input", type=str,
                                 help='root directory of input images, or path to single image')
        self.parser.add_argument("--threshold", type=float, default=0.5, help='threshold for accepting detection')
        self.parser.add_argument('--image-size', type=int, default=256,
                                 help='scale input image size instead of model defined size')
        self.parser.add_argument('-j', '--jobs', type=int, help='number of processes', default=1)
        self.parser.add_argument('--keep-ratio', type=lambda x: (str(x).lower() == 'true'), default=None,
                                 help='keep image ratio after scaling, otherwise squared')
        #self.parser.add_argument('--tta', type=str, default='', help='Test Time Augementation: orig,hflip,vflip,dflip,autocontrast,equalize')
        self.parser.add_argument('--tda', action='store_true',
                                 help="augment test dataset with flips")

        self.parser.add_argument('--eval', action='store_true', help='evaluation after detection')
        self.parser.add_argument('--export', type=str, help='format for exporting submission')

    def add_argument(self, *kargs, **kwargs):
        self.parser.add_argument(*kargs, **kwargs)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    def parse_args(self, args=None, namespace=None):
        args = self.parser.parse_args(args=args, namespace=namespace)

        if not args.output and not args.eval and not args.export:
            raise UserWarning("Please specify at least one path for output / evaluation / export")

        return args


def tta_pre_process(inputs, tta):
    if tta & 1:
        inputs = inputs.flip(-1)
    if tta & 2:
        inputs = inputs.flip(-2)
    if tta & 4:
        inputs = inputs.transpose(-1, -2)
    return inputs


def tta_post_process(outputs, tta):
    if tta & 4:
        outputs = outputs.transpose(-1, -2)
    if tta & 2:
        outputs = outputs.flip(-2)
    if tta & 1:
        outputs = outputs.flip(-1)
    return outputs


def predict_tta(model, inputs, tile_size, tta):
    inputs = tta_pre_process(inputs, tta)
    if tile_size > 0:
        five_tile = NineTile((tile_size, tile_size))  # 768
        outputs = five_tile.apply(model, inputs)
    else:
        outputs = model(inputs)
    outputs = tta_post_process(outputs, tta)
    probs = torch.sigmoid(outputs)
    return probs


def predict(samples, model, use_cuda, padding=False, tile_size=416, tta=0, tta_threshold=-1):
    inputs = samples['input']
    image_size = samples['image_size']

    if use_cuda:
        inputs = inputs.cuda()
    inputs = inputs.unsqueeze(0)

    if tta >= 0:
        probs = predict_tta(model, inputs, tile_size, tta)
    elif tta_threshold > 0:
        probs = 0
        n = 0
        for i in range(8):
            probs += predict_tta(model, inputs, tile_size, i)
            n += 1
            if (probs < tta_threshold * n).all():
                break
        probs /= n
    else:
        probs = [predict_tta(model, inputs, tile_size, t) for t in range(8)]
        probs = torch.cat(probs, dim=0)
        probs = torch.mean(probs, dim=0, keepdim=True)
    
    if padding:
        height, width = image_size
        assert probs.size(-2) >= height
        assert probs.size(-1) >= width
        probs = probs[..., :height, :width]

    return probs.cpu(), image_size


class FP16Model(torch.nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = network.half()

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


def create_model(model_file, use_cuda, half_float=False, padding=False, tile_size=416, tta=0, tta_threshold=-1):
    print('loading model', use_cuda, half_float, padding, tile_size)
    model = load_model(model_file)
    model.float()
    print('loaded')
    if use_cuda:
        model.cuda()
        if half_float:
            model = FP16Model(model)

    model.eval()

    ret = functools.partial(predict, model=model, use_cuda=use_cuda, padding=padding,
                            tile_size=tile_size, tta=tta, tta_threshold=tta_threshold)
    return ret, (1, 2, 3, 4, 5)


def pred_to_probs(pred):
    probs, image_size = pred
    probs = interpolate(probs.float(), size=image_size).squeeze().numpy()
    return probs


def probs_to_mask(probs, threshold=0.5):
    return (probs > threshold).astype(np.uint8)


def pred_to_mask(pred, threshold=0.5):
    probs = pred_to_probs(pred)
    return probs_to_mask(probs, threshold)


def probs_to_confidence(probs):
    return (np.abs(probs - 0.5) * 2).min(axis=0)


if __name__ == '__main__':
    import torch
    import json
    from data.xray import create_xray_dataset
    from train import test_transform, expand_valid_dataset
    from trainer.data import TransformedDataset, ImageFolder, ShuffledDataset

    parser = ArgumentParser()
    parser.add_argument('--data-root', default='', type=str, help='path to dataset of Jinnan xray Challenge')
    parser.add_argument('--data-fold', default=None, type=int, help='data fold id for cross validation')
    parser.add_argument('--data-norm', action='store_true',
                       help="use dataset normalization values instead of imagenet's")
    parser.add_argument('--data-normal', default=0, type=float,
                       help="percentage of normal data in train dataset")
    parser.add_argument('--data-shuffle', action='store_true', help="shuffle the dataset")
    parser.add_argument('--tile-size', default=0, type=int, help="the size of tile input")
    parser.add_argument('--tta', default=0, type=int, help="flag for TTA (0 ~ 7)")
    parser.add_argument('--tta-threshold', default=-1, type=float, help="threshold for suspend tta")
    args = parser.parse_args()

    if args.data_root:
        assert not args.input
        dataset = create_xray_dataset(args.data_root, mode='test', data_fold=args.data_fold)
        if args.tda:
            dataset = expand_valid_dataset(dataset)
    else:
        dataset = ImageFolder(args.input)

    if args.data_shuffle:
        dataset = ShuffledDataset(dataset)

    dataset_t = TransformedDataset(dataset, test_transform(args.image_size))

    print(dataset_t)
    the_create_model = functools.partial(create_model, half_float=args.half,
                                         padding=(args.image_size<=0),
                                         tile_size=args.tile_size,
                                         tta=args.tta,
                                         tta_threshold=args.tta_threshold)

    tester = Tester(create_model=the_create_model, device=args.device, jobs=args.jobs, disable_tqdm=False)
    predictions = tester.test(args.model, dataset_t, args.output)

    if args.export:
        from export import export
        export(predictions, args.export)

    if args.eval:
        if args.data_root:
            from eval import evaluate
            iou = evaluate(predictions, dataset)
            print(json.dumps(iou, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False))
        else:
            warnings.warn("ignore eval arg")
