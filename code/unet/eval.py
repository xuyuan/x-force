import numpy as np
from test import pred_to_mask
from tqdm import tqdm

from data.coco import mask_to_class_segments


def compute_inter_and_union(pred, target):
    if pred.ndim == 2:
        pred = np.asarray(list(mask_to_class_segments(pred, 5)))
    target_mask = np.asarray(list(mask_to_class_segments(target, len(pred))))
    return target_mask, pred & target_mask, pred | target_mask


class Metric(object):
    def __init__(self, classnames):
        self.classnames = classnames
        n = len(classnames)
        self.inter = np.zeros(n)
        self.union = np.zeros(n)
        self.pred = np.zeros(n)
        self.target = np.zeros(n)

    def update(self, pred, target):
        t, i, u = compute_inter_and_union(pred, target)
        self.inter += np.sum(np.sum(i, axis=-1), axis=-1)
        self.union += np.sum(np.sum(u, axis=-1), axis=-1)
        self.pred += np.sum(np.sum(pred, axis=-1), axis=-1)
        self.target += np.sum(np.sum(t, axis=-1), axis=-1)

    def iou(self):
        return self.inter / self.union

    def accuracy(self):
        return self.inter / self.pred

    def recall(self):
        return self.inter / self.target

    def summary(self):
        iou = self.iou()
        acc = self.accuracy()
        recall = self.recall()
        ret = {k: {'score': v, 'accuracy': a, 'recall': r}
               for k, v, a, r in zip(self.classnames, iou, acc, recall)}

        mean = np.mean(iou)
        m_acc = np.mean(acc)
        m_recall = np.mean(recall)
        ret['mean'] = {'score': mean, 'accuracy': m_acc, 'recall': m_recall}
        ret['score'] = mean
        return ret

    def __repr__(self):
        s = self.summary()
        return json.dumps(s, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)


def evaluate(predictions, dataset, num_processes=1):
    metric = Metric(dataset.classnames[1:])
    for sample in tqdm(dataset, disable=None):
        image_id = sample['image_id']
        pred = predictions[image_id]
        mask = pred_to_mask(pred)
        metric.update(mask, sample['masks'])

    return metric.summary()


def evaluate_submission(predictions, dataset, num_processes=1):
    metric = Metric(dataset.classnames[1:])
    for sample in tqdm(dataset, disable=None):
        image_id = sample['image_id']
        mask = predictions[image_id]['masks']
        metric.update(mask, sample['masks'])

    return metric.summary()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json
    from torch.utils.data import RandomSampler
    from trainer import Predictions
    from data.xray import create_xray_dataset
    from data.pseudolabel import PseudoLabel

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument('--data-root', default='', type=str, help='path to dataset of Jinnan xray Challenge')
    parser.add_argument('--data-fold', default=None, type=int, help='data fold id for cross validation')
    parser.add_argument('--data-normal', default=0, type=float,
                        help="percentage of normal data in train dataset")
    parser.add_argument('--data-pseudolabel', type=str, help="path to annotation of pseudolabel")
    parser.add_argument('predictions', type=str, help='pickle file of inference results or submission json file')
    args = parser.parse_args()

    if args.data_root:
        assert not args.data_pseudolabel
        dataset = create_xray_dataset(args.data_root, mode='test', data_fold=args.data_fold)
    elif args.data_pseudolabel:
        assert not args.data_root
        dataset = PseudoLabel(None, args.data_pseudolabel)
    print(dataset)

    pred_ext = Path(args.predictions).suffix
    if pred_ext == '.pkl':
        predictions = Predictions.open(args.predictions)
        iou = evaluate(predictions, dataset)
    elif pred_ext == '.json':
        predictions = PseudoLabel(None, args.predictions)
        iou = evaluate_submission(predictions, dataset)
    else:
        raise RuntimeError(f'unknown format {pred_ext}')

    print(json.dumps(iou, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False))
