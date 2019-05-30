
import json
from pathlib import Path
import numpy as np
from PIL import Image
from trainer.data import Dataset
from pycocotools.mask import decode
from nn import xray_classnames


class PseudoLabel(Dataset):
    classnames = ['background'] + list(xray_classnames)


    def __init__(self, images_root, annotations):
        if images_root:
            self.root = Path(images_root)
        else:
            self.root = None
        self.annotations = json.load(open(annotations))
        self.image_ids = list(self.annotations.keys())

    def __getitem__(self, index):
        if isinstance(index, str):# and index.endswith('.jpg'):
            image_id = index
        else:
            image_id = self.image_ids[index]

        anno = self.annotations[image_id]
        height, width = anno['size']
        masks = np.zeros((height, width), dtype=np.int32)
        for cls_id, rle in anno['mask'].items():
            fortran_mask = decode(rle)
            c_mask = np.ascontiguousarray(fortran_mask)
            cls_id = int(cls_id)
            l = (2 ** (31 - cls_id)) | (2 ** cls_id)  # 1 instance per class
            masks = masks | (c_mask * l).astype(np.int32)
        sample = dict(image_id=image_id, masks=masks)

        if self.root:
            image_file = self.root / image_id
            sample['image'] = Image.open(image_file)

        return sample

    def __len__(self):
        return len(self.annotations)

    def __repr__(self):
        fmt_str = self.__class__.__name__
        fmt_str += f'\n    images_root:{self.root}'
        fmt_str += f'\n    len:{len(self)}\n'
        return fmt_str


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dataset = PseudoLabel(None, #'/media/data/jinnan2/jinnan2_round2_test_b_20190424',
                          '/media/data/jinnan2/submission_4402.json')
    dataset2 = PseudoLabel(None, #'/media/data/jinnan2/jinnan2_round2_test_b_20190424',
                          '/media/data/jinnan2/submission_4409.json')
    print(len(dataset))
    print(dataset)

    def show_sample(sample):
        if 'image' in sample:
            plt.imshow(sample['image'])
        if np.any(sample['masks']):
            plt.imshow(sample['masks'], alpha=0.7, cmap='tab20')

    for i, sample in enumerate(dataset):
        print(sample)
        plt.figure(figsize=(50, 20))

        sample2 = dataset2[i]
        plt.subplot(121)
        show_sample(sample)
        plt.subplot(122)
        plt.imshow(sample2['masks'] - sample['masks'])
        plt.show()
