

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from trainer.swa import swa
    from data.xray import create_xray_dataset
    from train import valid_transform, TransformedDataset
    import nn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, help='input directory which contains models')
    parser.add_argument("-o", "--output", type=str, default='swa_model.pth', help='output model file')
    parser.add_argument("--batch-size", type=int, default=8, help='batch size')
    parser.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')

    group = parser.add_argument_group('options of dataset')
    group.add_argument('--data-root', default='', type=str, help='path to dataset of Jinnan xray Challenge')
    group.add_argument('--data-fold', default=None, type=int, help='data fold id for cross validation')
    group.add_argument('--data-norm', action='store_true',
                       help="use dataset normalization values instead of imagenet's")
    group.add_argument('--data-normal', default=0, type=float,
                       help="percentage of normal data in train dataset")
    args = parser.parse_args()

    train_dataset = create_xray_dataset(args.data_root, mode='train', data_fold=args.data_fold, data_normal=args.data_normal)
    dataset = TransformedDataset(train_dataset, valid_transform(256))

    net = swa(nn.load, args.input, dataset, args.batch_size, args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    net.save(args.output)
