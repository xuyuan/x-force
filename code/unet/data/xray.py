from pathlib import Path
import random
import json
import os.path as osp
from PIL import Image, ImageChops
from .coco import COCODetection
from trainer.data import ImageFolder, RollSplitSet
import numpy as np


class XRayDataset(COCODetection):
    def __init__(self, images_root, annotations, image_ids=None,
                 return_image_file=False, mixup_normal_images=False,
                 sort_by_image_size=False, segmentation=True):
        super().__init__(images_root=images_root, annotations=annotations, dataset_name=None,
                         image_ids=image_ids, return_image_file=return_image_file,
                         sort_by_image_size=sort_by_image_size, segmentation=segmentation)
        if mixup_normal_images:
            mixup_normal_images_root = Path(self.root) / 'normal'
            self.mixup_normal_images = list(mixup_normal_images_root.iterdir())
        else:
            self.mixup_normal_images = None

        self.mixup_prob = 0.5
        self.mixup_alpha = 1.5

        # fix area for COCO eval
        for ann in self.coco.anns.values():
            bbox = ann['bbox']
            ann['area'] = bbox[2] * bbox[3]

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        if self.mixup_normal_images and random.random() < self.mixup_prob:
            normal_image = random.choice(self.mixup_normal_images)
            normal_sample = super().__getitem__(normal_image.name)

            weight = random.betavariate(self.mixup_alpha, self.mixup_alpha)
            if weight > 0.5:
                weight = 1 - weight

            image_a = sample['image']
            image_b = normal_sample['image']
            image_b = image_b.resize(image_a.size)
            image_ab = Image.blend(image_a, image_b, weight)

            sample['image'] = image_ab

        return sample

    def _get_image_path(self, img_id):
        if isinstance(img_id, str):
            return osp.join(self.root, 'normal', img_id)
        else:
            return osp.join(self.root, 'restricted', self.coco.loadImgs(img_id)[0]['file_name'])


class XrayNormalDataset(ImageFolder):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        width, height = sample['image'].size
        sample['masks'] = np.zeros((height, width), dtype=np.int32)
        return sample


def create_xray_dataset(xray_root, mode, data_fold=None, return_image_file=False,
                        sort_by_image_size=False, data_normal=0):
    xray_root = Path(xray_root)
    if not xray_root.exists():
        raise RuntimeError(f"{xray_root} not exists")

    xray_annotations = xray_root / 'train_restriction.json'
    if not xray_annotations.exists():
        raise RuntimeError(f"{xray_annotations} not exists")

    if data_fold is not None:

        folds = json.load(open(Path(__file__).parent / 'round2_5fold.json'))
        image_ids = folds[data_fold][mode]
    else:
        image_ids = None

    mixup_normal_images = mode == 'train'

    xray_dataset = XRayDataset(str(xray_root), str(xray_annotations),
                               image_ids=image_ids,
                               return_image_file=return_image_file,
                               mixup_normal_images=mixup_normal_images,
                               sort_by_image_size=sort_by_image_size)

    if mode == 'train' and data_normal > 0:
        normal_dataset = XrayNormalDataset(xray_root / 'normal')
        splits = int(len(normal_dataset) / len(xray_dataset) / data_normal)  # 30% normal images
        if splits > 1:
            normal_dataset = RollSplitSet(normal_dataset, splits)
        xray_dataset = xray_dataset + normal_dataset

    return xray_dataset


if __name__ == '__main__':
    import sys
    image_files = []
    dataset = create_xray_dataset(sys.argv[1], 'test', data_fold=sys.argv[2], return_image_file=True)[0]
    for i in dataset.ids:
        if isinstance(i, str):
            image_files.append('normal/'+i)
        else:
            image_files.append('restricted/' + Path(dataset._get_image_path(i)).name)

    print(json.dumps(image_files))
