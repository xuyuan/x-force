

import warnings
from functools import cmp_to_key
import os.path as osp
import numpy as np
from PIL import Image
from skimage import feature
from trainer.data import Dataset

COCO_LABELS = '''
1,1,person
2,2,bicycle
3,3,car
4,4,motorcycle
5,5,airplane
6,6,bus
7,7,train
8,8,truck
9,9,boat
10,10,traffic light
11,11,fire hydrant
13,12,stop sign
14,13,parking meter
15,14,bench
16,15,bird
17,16,cat
18,17,dog
19,18,horse
20,19,sheep
21,20,cow
22,21,elephant
23,22,bear
24,23,zebra
25,24,giraffe
27,25,backpack
28,26,umbrella
31,27,handbag
32,28,tie
33,29,suitcase
34,30,frisbee
35,31,skis
36,32,snowboard
37,33,sports ball
38,34,kite
39,35,baseball bat
40,36,baseball glove
41,37,skateboard
42,38,surfboard
43,39,tennis racket
44,40,bottle
46,41,wine glass
47,42,cup
48,43,fork
49,44,knife
50,45,spoon
51,46,bowl
52,47,banana
53,48,apple
54,49,sandwich
55,50,orange
56,51,broccoli
57,52,carrot
58,53,hot dog
59,54,pizza
60,55,donut
61,56,cake
62,57,chair
63,58,couch
64,59,potted plant
65,60,bed
67,61,dining table
70,62,toilet
72,63,tv
73,64,laptop
74,65,mouse
75,66,remote
76,67,keyboard
77,68,cell phone
78,69,microwave
79,70,oven
80,71,toaster
81,72,sink
82,73,refrigerator
84,74,book
85,75,clock
86,76,vase
87,77,scissors
88,78,teddy bear
89,79,hair drier
90,80,toothbrush
'''


def mask_to_class_segments(masks, n_class=5):
    for i in range(n_class):
        yield (masks & (2 ** (30-i))) > 0
        #yield (masks & (2 ** i)) > 0


def mask_to_instances(masks):
    for i in range(30-5):
        instance = (masks & (2 ** i))
        if instance.any():
            yield instance


def mask_to_instance_edges(masks):
    for i in mask_to_instances(masks):
        yield feature.canny(i > 0, sigma=0).astype(np.int32)


class InstanceEdgeTrans(object):
    def __call__(self, sample):
        masks = sample['masks']
        sample['edges'] = masks * (sum(mask_to_instance_edges(masks)) > 0)
        return sample


def get_label_map():
    label_map = {}
    classnames = ['background']
    for line in COCO_LABELS.splitlines():
        ids = line.split(',')
        if len(ids) >= 3:
            label_map[int(ids[0])] = int(ids[1])
            classnames.append(ids[2])
    return label_map, classnames


class COCOSegmentationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initialized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, coco, label_map=None):
        self.coco = coco
        self.label_map = label_map
        self.num_cats = len(coco.cats)

    def __call__(self, target, img):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
            rotation (float): angle of rotation, minAreaRect label is required
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """

        width, height = img.size

        masks = np.zeros((height, width), dtype=np.int32)

        for i, obj in enumerate(target):
            if not i + self.num_cats < 31:
                #warnings.warn(f"number of objects {i} is more than {31 - self.num_cats}")
                i = i % (31 - self.num_cats)
            label_idx = obj['category_id']
            if self.label_map:
                label_idx = self.label_map[label_idx]

            if 'segmentation' in obj:
                l = (2 ** (31 - label_idx)) | (2 ** i)
                #l = (2 ** (label_idx - 1))
                masks = masks | (self.coco.annToMask(obj) * l).astype(np.int32)
            else:
                warnings.warn("no segmentation!")

        sample = dict(image=img, masks=masks)
        return sample


class COCODetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2014>`_ Dataset.
    Args:
        images_root (string): Root directory where images are downloaded to.
        annotations (string): annotations file of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """
    metric = 'coco'

    def __init__(self, images_root, annotations, dataset_name=None, image_ids=None,
                 random_rotation=0, return_image_file=False, sort_by_image_size=False,
                 segmentation=False):
        assert random_rotation == 0
        from pycocotools.coco import COCO
        self.return_image_file = return_image_file
        self.root = images_root
        self.coco = COCO(annotations)
        if image_ids is None:
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            self.ids = image_ids

        label_map = None
        self.classnames = None
        if dataset_name == 'MS COCO':
            label_map, self.classnames = get_label_map()

        assert segmentation
        self.target_transform = COCOSegmentationTransform(self.coco, label_map)
        if self.classnames is None:
            cats = self.coco.cats
            self.classnames = ['background'] * (len(cats) + 1)
            for c in cats.values():
                self.classnames[c['id']] = c['name']

        if sort_by_image_size:
            self.ids = sorted(self.ids, key=cmp_to_key(self._image_size_compare))

    def _image_size_compare(self, x, y):
        x = self._read_image(x)
        y = self._read_image(y)
        if x.height == y.height:
            return x.width - y.width
        return x.height - y.height

    def __len__(self):
        return len(self.ids)

    def _get_image_path(self, img_id):
        if isinstance(img_id, str):
            return osp.join(self.root, img_id)
        else:
            return osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])

    def _read_image(self, img_id):
        path = self._get_image_path(img_id)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        return Image.open(osp.join(self.root, path))

    def __getitem__(self, index):
        if isinstance(index, str):
            img_id = index
        else:
            img_id = self.ids[index]

        if isinstance(img_id, str):
            target = []  # no objects
        else:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)

        img = self._read_image(img_id)
        sample = dict(image=img, image_id=str(img_id))
        if self.target_transform is not None:
            target = self.target_transform(target, img)
            sample.update(target)

        if self.return_image_file:
            sample['image_file'] = self._get_image_path(img_id)

        return sample

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'

        for key, value in self.coco.dataset['info'].items():
            fmt_str += f'    {key}: {value}\n'
        fmt_str += '    Number of datapoints: {}\n'.format(len(self))
        fmt_str += '    Number of classes: {}\n'.format(len(self.classnames))
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def loadRes(dataset, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        from pycocotools.coco import COCO
        import json
        import time
        import copy
        from pycocotools import mask as maskUtils
        self = dataset.coco

        res = COCO()
        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == bytes:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = set([ann['image_id'] for ann in anns])
        gtImgIds = set([str(i) for i in set(self.getImgIds())])
        validImgIds = annsImgIds & gtImgIds
        if len(validImgIds) < len(annsImgIds):
            print(f'skip {len(annsImgIds) - len(validImgIds)} images which does not have annotation')
            anns = [ann for ann in anns if ann['image_id'] in validImgIds]

        annsImgIds = []
        for ann in anns:
            image_id = int(ann['image_id'])  # str -> int for COCO
            ann['image_id'] = image_id
            annsImgIds.append(image_id)
        res.dataset['images'] = self.loadImgs(self.getImgIds(annsImgIds))

        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

            # map class_id to category_id in coco api
            category_id_map = None
            if dataset.target_transform.label_map:
                category_id_map = {v: k for k, v in dataset.target_transform.label_map.items()}

            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
                if category_id_map:
                    ann['category_id'] = category_id_map[ann['category_id']]
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


class COCOSegmentation(COCODetection):
    def __init__(self, images_root, annotations, transform=None, dataset_name=None, image_ids=None,
                 random_rotation=0, return_image_file=False, sort_by_image_size=False):
        super().__init__(images_root=images_root, annotations=annotations, transform=transform,
                         dataset_name=dataset_name, image_ids=image_ids,
                         random_rotation=random_rotation, return_image_file=return_image_file,
                         sort_by_image_size=sort_by_image_size)
        self.target_transform.segmentation = True


if __name__ == '__main__':
    dataset = COCOSegmentation('/media/data/coco',
                               '/media/data/coco/annotations/instances_val2017.json',
                               dataset_name='MS COCO')
    print(dataset)
    print(dataset.classnames)
    sample = dataset[0]
    print(sample)
    #sample['image'].show()
