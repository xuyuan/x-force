
import numpy as np
import matplotlib.pyplot as plt
from eval import Metric
from nn import xray_classnames
from test import probs_to_mask, pred_to_probs, probs_to_confidence


def merge_mask(mask):
    out = np.zeros(mask.shape[1:], dtype=np.int)
    for i, m in enumerate(mask):
        out |= (m * (2 ** (31 - i)))
    return out


def show_sample_and_mask(sample, masks=None, probs=None):
    plt.figure(figsize=(50, 20))
    plt.subplot(2, 3, 1)
    plt.imshow(sample['image'])
    plt.title(sample['image_id'])
    #print(sample)
    cmap = 'tab20'
    plt.subplot(2, 3, 3)
    plt.imshow(sample['masks'], cmap=cmap)
    plt.title('ground truth')

    plt.subplot(2, 3, 2)
    if masks is None:
        plt.imshow(sample['image'])
        plt.imshow(sample['masks'], alpha=0.7, cmap=cmap)
    else:
        mask = merge_mask(masks)
        plt.imshow(mask, cmap=cmap)
        plt.title('prediction')

    plt.subplot(2, 3, 4)
    if masks is not None:
        plt.imshow((sample['masks'] - mask), cmap=cmap)
        metric = Metric(xray_classnames)
        metric.update(masks, sample['masks'])
        plt.title(str(metric.iou()))

    if probs is not None:
        plt.subplot(2, 3, 6)
        plt.imshow(probs.max(axis=0))

        confidence = probs_to_confidence(probs)
        plt.subplot(2, 3, 5)
        plt.imshow(confidence)
        confidence = np.mean(confidence)
        plt.title(str(confidence))

    plt.show()


def vis(sample, predictions):
    masks = None
    pred = None
    probs = None
    if predictions:
        image_id = sample['image_id']
        pred = predictions[image_id]
        probs = pred_to_probs(pred)
        masks = probs_to_mask(probs)

    show_sample_and_mask(sample, masks, probs)


if __name__ == "__main__":
    import argparse
    from torch.utils.data import RandomSampler
    from trainer import Predictions
    from data.xray import create_xray_dataset

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument('--data-root', default='', type=str, help='path to dataset of Jinnan xray Challenge')
    parser.add_argument('--data-fold', default=None, type=int, help='data fold id for cross validation')
    parser.add_argument('--predictions', type=str, help='pickle file of inference results')
    parser.add_argument('--sample', type=int, help='show given sample')
    args = parser.parse_args()

    dataset = create_xray_dataset(args.data_root, mode='test', data_fold=args.data_fold)
    print(dataset)

    predictions = None
    if args.predictions:
        predictions = Predictions.open(args.predictions)

    if args.sample is not None:
        sample = dataset[args.sample]
        vis(sample, predictions)
    else:
        sampler = RandomSampler(dataset)
        for i in sampler:
            sample = dataset[i]
            vis(sample, predictions)
