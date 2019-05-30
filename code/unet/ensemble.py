
from pathlib import Path
from tqdm import tqdm


def ensemble(detection_a, detection_b, weight, out, process_id=None, use_cuda=False):
    desc = "ensemble"
    if process_id is not None:
        desc += " #{}".format(process_id)
    pbar = tqdm(detection_a, unit="images", desc=desc, position=process_id, disable=None)

    for image_id, (probs_a, image_size_a) in pbar:
        probs_b, image_size_b = detection_b[image_id]
        assert image_size_a == image_size_b
        if use_cuda:
            probs_a = probs_a.cuda()
            probs_b = probs_b.cuda()
        probs = (probs_a + probs_b * weight) / (1 + weight)
        if use_cuda:
            probs = probs.cpu()
        out[image_id] = (probs, image_size_a)


def ensemble_all(all_predictions, weights, out, use_cuda=False):
    detection_a = all_predictions[0]
    weight_a = weights[0]
    for b in range(1, len(all_predictions)):
        weight = args.weights[b] / weight_a
        ensemble(detection_a, all_predictions[b], weight, out=out, use_cuda=use_cuda)
        weight_a += args.weights[b]
        detection_a = out


def find_prediction_file(path):
    path = Path(path)
    if path.is_dir():
        return list(path.glob("**/detections.pkl"))
    else:
        return [path]


if __name__ == '__main__':
    import argparse
    from trainer import Predictions

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('predictions', type=str, nargs='+', help='pickle file of prediction results')
    parser.add_argument('-o', '--out', type=str, help='output directory of ensembled results')
    parser.add_argument('-f', '--force', action='store_true', help='remove output directory if it exists')
    parser.add_argument('-w', '--weights', type=float, nargs='+', help='weights for detection results')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=args.force)

    prediction_files = []
    for f in args.predictions:
        prediction_files += find_prediction_file(f)
    print(f'input prediction files: {prediction_files}')

    all_predictions = [Predictions.open(filename=f) for f in prediction_files]
    assert len(all_predictions) > 1
    if args.weights is None or len(args.weights) == 0:
        args.weights = [1] * len(all_predictions)
    assert len(args.weights) == len(all_predictions)

    out = Predictions.open(str(out_dir / 'detections.pkl'), mode='w', n_new_file=1, classnames=all_predictions[0].classnames)

    ensemble_all(all_predictions, args.weights, out, use_cuda=args.cuda)
