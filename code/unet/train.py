
import functools
from pathlib import Path
import shutil

from torch.nn.functional import binary_cross_entropy_with_logits
from trainer import Trainer
from trainer.transforms import *
from trainer.transforms.vision import *
from trainer.test import inference
from trainer.utils import get_num_workers
from trainer.predictions import Predictions
from trainer.data import split_dataset, TransformedDataset
from test import predict
from eval import evaluate
from data.coco import InstanceEdgeTrans

WHITE = (255, 255, 255)

## from round1_test_a
#mean=[0.80865544 0.82481456 0.72233427], std=[0.29703274 0.21022193 0.2871294 ]

## from round2_test_a
mean = [0.83288646, 0.837719,   0.73465323]
std = [0.2761646,  0.20110638, 0.2877356 ]


def train_transform(size=256, img_norm=TORCH_VISION_NORMALIZE):
    return Compose([FilterSample(['image', 'masks']),
                    RandomOrder([RandomApply([RandomPad(max_padding=size//3, fill=WHITE)]),
                                 RandomApply([TryApply(RandomCrop(size//3*2, size*3//2))]),
                                 ]),
                    Resize(size),
                    RandomOrder([RandomApply([HorizontalFlip()]),
                                 RandomApply([VerticalFlip()]),
                                 RandomApply([Transpose()]),
                                 RandomApply([RandmonRotate(22.5, fill=WHITE)]),

                                 # color
                                 RandomChoice([ColorJitter(brightness=0.125, contrast=0.5,
                                                           saturation=0.5, hue=0.05),
                                               RandomApply([AutoContrast()])]),

                                 # noise
                                 RandomApply([RandomChoice([SaltAndPepper(),
                                                            RandomSharpness(),
                                                            JpegCompression()])]),

                                 # deform
                                 RandomApply([RandomChoice([
                                                            RandomHorizontalShear(0.3),
                                                            RandomSkew(0.3),
                                                            GridDistortion(),
                                                           ])])
                                 ] + [RandomApply([TryApply(CutOut(fill=WHITE, max_size=size//4))])] * 3
                                ),
                    InstanceEdgeTrans(),
                    ToTensor(img_norm)])


def valid_transform(size=256, img_norm=TORCH_VISION_NORMALIZE):
    return Compose([FilterSample(['image', 'masks']),
                    Resize(size),
                    ToTensor(img_norm)])


def append_image_id(sample, s):
    return {k: v + s if k == 'image_id' else v for k, v in sample.items()}


def ImageIdAppend(s):
    return tvt.Lambda(functools.partial(append_image_id, s=s))


def expand_valid_dataset(dataset):
    expand = TransformedDataset(dataset, Compose([HorizontalFlip(), ImageIdAppend('H')]))
    dataset = dataset + expand

    expand = TransformedDataset(dataset, Compose([VerticalFlip(), ImageIdAppend('V')]))
    dataset = dataset + expand

    expand = TransformedDataset(dataset, Compose([Transpose(), ImageIdAppend('T')]))
    dataset = dataset + expand

    return dataset


def test_transform(size=256, img_norm=TORCH_VISION_NORMALIZE):
    trans = [FilterSample(['image', 'masks', 'image_id']),
             RecordImageSize()]
    if size > 0:
        image_trans = Resize(size)
    else:
        image_trans = DivisiblePad(64, fill=WHITE)

    trans += [ApplyOnly(['image'], image_trans),
              ToTensor(img_norm)]

    return Compose(trans)


class IoUTrainer(Trainer):
    def test(self):
        """
        TODO: for abstracting in trainer
        * forward function should be cuda aware by itself
        * or pre_test (should be in transform?) and post_test for processing

        * classnames should be abstract to sth. common, e.g. targets? target_names?
        * or can be ignored here: just let's evaluation assume it is the same between dataset
        """
        test_dataset = self.datasets['test']
        num_workers = get_num_workers(self.args.jobs)

        forward = functools.partial(predict, model=self.net, use_cuda=self.use_cuda)

        if args.local_rank is None:
            detections, inference_time = inference(forward, test_dataset)
            predictions = Predictions([detections])
            evaluation = evaluate(predictions, test_dataset.dataset, num_processes=num_workers)
        else:
            # distributed test
            world_size = torch.distributed.get_world_size()
            sub_test_dataset = split_dataset(test_dataset, world_size)[args.local_rank]
            output_root = Path(self.args.log_dir) / "test"
            if args.local_rank == 0:
                shutil.rmtree(output_root, ignore_errors=True)
                output_root.mkdir(parents=True)
            torch.distributed.barrier()  # synchronizes all processes.
            detections, inference_time = inference(forward, sub_test_dataset, output_root=output_root, process_id=args.local_rank)
            torch.distributed.barrier() # synchronizes all processes.
            if args.local_rank == 0:
                # evaluation in main process only
                detection_files = [str(f)[:-4] for f in output_root.glob('*.dir')]
                predictions = Predictions(detection_files)
                evaluation = evaluate(predictions, test_dataset.dataset, num_processes=num_workers)
            else:
                evaluation = None
            torch.distributed.barrier()  # synchronizes all processes.

        return evaluation


if __name__ == '__main__':
    from data.xray import create_xray_dataset, RollSplitSet
    from data.pseudolabel import PseudoLabel
    from trainer import ArgumentParser
    import nn
    from loss import *

    parser = ArgumentParser()
    group = parser.add_argument_group('options of dataset')
    group.add_argument('--data-root', default='', type=str, help='path to dataset of Jinnan xray Challenge')
    group.add_argument('--data-fold', default=None, type=int, help='data fold id for cross validation')
    group.add_argument('--data-norm', action='store_true',
                       help="use dataset normalization values instead of imagenet's")
    group.add_argument('--data-normal', default=0, type=float,
                       help="percentage of normal data in train dataset")
    group.add_argument('--data-pseudolabel-images', type=str, help="path to images of pseudolabel")
    group.add_argument('--data-pseudolabel-labels', type=str, help="path to annotation of pseudolabel")
    group.add_argument('--data-pseudolabel-images2', type=str, help="path to images of pseudolabel")
    group.add_argument('--data-pseudolabel-labels2', type=str, help="path to annotation of pseudolabel")
    group.add_argument('--data-pseudolabel-rate', default=0, type=float, help="percentage of pseudolabel in train dataset")
    group.add_argument('--image-size', default=256, type=int,
                       help="input image size")
    group.add_argument('--test-image-size', default=0, type=int,
                       help="input image size for validation and test")

    group = parser.add_argument_group('options of model')
    group.add_argument('--version', default='unet', help='model version', type=str)
    group.add_argument('--model', default='', help='load saved model')
    group.add_argument('--basenet', default='resnet50', choices=nn.BASENET_CHOICES, help='pretrained basenet')
    group.add_argument('--weights', type=str, help='pretrained weights')

    group = parser.add_argument_group('options of criterion')
    group.add_argument("--loss", default='bce', help='loss function of NN',
                        choices=['bce', 'focal', 'robust_focal', 'lovasz', 'lovasz_bce', 'lovasz_focal', 'soft_jaccard',
                                 'lovasz_symmetric', 'lovasz_focal_symmetric', 'dice', 'dice_and_bce'])

    args = parser.parse_args()

    image_size = args.image_size
    test_image_size = args.test_image_size if args.test_image_size > 0 else image_size
    img_norm = tvt.Normalize(mean=mean, std=std) if args.data_norm else TORCH_VISION_NORMALIZE
    data_augs = dict(train=train_transform(image_size, img_norm),
                     valid=valid_transform(test_image_size, img_norm),
                     test=test_transform(test_image_size, img_norm),
                     )

    datasets = {mode: create_xray_dataset(args.data_root, mode=mode, data_fold=args.data_fold,
                                          data_normal=args.data_normal)
                for mode in data_augs}

    if args.data_pseudolabel_rate > 0:
        assert args.data_pseudolabel_images
        assert args.data_pseudolabel_labels
        for pseudolabel_images, pseudolabel_labels in ((args.data_pseudolabel_images, args.data_pseudolabel_labels),
                                                       (args.data_pseudolabel_images2, args.data_pseudolabel_labels2)):
            plabel = PseudoLabel(pseudolabel_images, pseudolabel_labels)
            splits = int(len(plabel) / len(datasets['train']) / args.data_pseudolabel_rate)  # 30% normal images
            if splits > 1:
                plabel = RollSplitSet(plabel, splits)
            datasets['train'] += plabel

    #datasets = {mode: d if mode == 'train' else expand_valid_dataset(d) for mode, d in datasets.items()}

    datasets_t = {mode: TransformedDataset(d, data_augs[mode]) for mode, d in datasets.items()}
    classnames = datasets['train'].classnames[1:]

    if args.model:
        print('loading model {}...'.format(args.model))
        model = nn.load(args.model)
    else:
        model = nn.create(args.version, args.basenet, classnames)
        if args.weights:
            print(f'loading weights {args.weights}')
            weights = torch.load(args.weights, map_location='cpu')
            weights = weights.get('state_dict', weights)
            model.load_state_dict(weights, strict=True)

    loss_functions = dict(bce=binary_cross_entropy_with_logits,
                          focal=binary_focal_loss_with_logits,
                          lovasz=lovasz_hinge,
                          lovasz_bce=lovasz_bce_with_logits,
                          lovasz_focal_symmetric=lovasz_focal_symmetric,
                          dice=dice_loss_with_logits,
                          dice_and_bce=dice_and_bce)
    loss_func = loss_functions[args.loss]

    the_criterion = functools.partial(model.criterion, loss_func=loss_func)

    trainer = IoUTrainer(model, datasets_t, the_criterion, args)
    trainer.run()
