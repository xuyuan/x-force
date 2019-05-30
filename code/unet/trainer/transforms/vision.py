
import warnings
from io import BytesIO
import collections
import numbers

import numpy as np
from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageDraw
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as tvt
import skimage
from skimage.exposure import equalize_adapthist
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import cv2
from .common import RandomChoice, Compose, RandomApply, PassThough


_pil_interpolation_to_str = {
    Image.NEAREST: 'NEAREST',
    Image.BILINEAR: 'BILINEAR',
    Image.BICUBIC: 'BICUBIC',
    Image.LANCZOS: 'LANCZOS',
}


def cv2_border_mode_value(border):
    if border == 'replicate':
        border_mode = cv2.BORDER_REFLECT_101
        border_value = 0
    else:
        border_mode = cv2.BORDER_CONSTANT
        border_value = border
    return dict(borderMode=border_mode, borderValue=border_value)


# normalization for `torchvision.models`
# see http://pytorch.org/docs/master/torchvision/models.html
TORCH_VISION_MEAN = np.asarray([0.485, 0.456, 0.406])
TORCH_VISION_STD = np.asarray([0.229, 0.224, 0.225])
TORCH_VISION_NORMALIZE = tvt.Normalize(mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD)
TORCH_VISION_DENORMALIZE = tvt.Normalize(mean=-TORCH_VISION_MEAN/TORCH_VISION_STD, std=1/TORCH_VISION_STD)


def _random(name, data=(-1, 1)):
    if name == 'choice':
        return np.random.choice(data)
    elif name == 'uniform':
        return np.random.uniform(data[0], data[1])
    else:
        raise NotImplementedError(name)


class VisionTransform(object):
    def __repr__(self): return self.__class__.__name__ + '()'

    def __call__(self, sample):
        """
        :param sample: dict of data, key is used to determine data type, e.g. image, bbox, mask
        :return: transformed sample in dict
        """
        sample = self.pre_transform(sample)
        output_sample = {}
        for k, v in sample.items():
            if k == 'image':
                output_sample[k] = self.transform_image(v)
            elif k == 'bbox':
                output_sample[k] = self.transform_bbox(v) if len(v) > 0 else v
            elif k.startswith('mask'):
                output_sample[k] = self.transform_mask(v)
            else:
                output_sample[k] = sample[k]

        output_sample = self.post_transform(output_sample)
        return output_sample

    def pre_transform(self, sample):
        return sample

    def transform_image(self, image):
        return image

    def transform_bbox(self, bbox):
        return bbox

    def transform_mask(self, mask):
        return mask

    def post_transform(self, sample):
        # if 'image' in sample:
        #     w, h = sample['image'].size
        #     for k, v in sample.items():
        #         if k.startswith('mask'):
        #             if v.shape[0] != h or v.shape[1] != w:
        #                 raise RuntimeError(f'{repr(self)}\n mask size mismatch {(h, w)} != {(v.shape)}')
        return sample

    @staticmethod
    def size_from_number_or_iterable(size):
        if isinstance(size, numbers.Number):
            return (int(size), int(size))
        elif isinstance(size, collections.Iterable):
            return size
        else:
            raise RuntimeError(type(size))


class Resize(VisionTransform):
    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        :param size: Desired output size (h, w)
        :param interpolation: interpolation method of PIL
        """
        self.size = self.size_from_number_or_iterable(size)
        self.interpolation = interpolation

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

    def transform_image(self, image):
        return F.resize(image, self.size)

    def transform_mask(self, mask):
        return skimage.transform.resize(mask, self.size,
                                        order=0, preserve_range=True,
                                        mode='constant', anti_aliasing=False
                                        ).astype(mask.dtype)


class RecordImageSize(VisionTransform):
    def pre_transform(self, sample):
        image = sample['image']
        sample['image_size'] = (image.height, image.width)
        return sample


class Pad(VisionTransform):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1})'.format(self.padding, self.fill)

    def pre_transform(self, sample):
        image = sample['image']
        self.image_size = image.size
        return sample

    def transform_image(self, image):
        return F.pad(image, self.padding, fill=self.fill)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            left = self.padding[0] / self.image_size[0]
            top = self.padding[1] / self.image_size[1]
            bbox[:, :2] += (int(left), int(top))
            bbox[:, 2:] += (int(left), int(top))
        return bbox

    def transform_mask(self, mask):
        left, top, right, bottom = self.padding
        return np.pad(mask, ((top, bottom), (left, right)), 'constant')


class DivisiblePad(Pad):
    def __init__(self, divisible, fill=0):
        super().__init__(0, fill)
        self.divisible = divisible

    def __repr__(self):
        return self.__class__.__name__ + '(divisible={0}, fill={1})'.format(self.divisible, self.fill)

    def pre_transform(self, sample):
        sample = super().pre_transform(sample)
        width, height = self.image_size
        right = self.divisible_padding(width)
        bottom = self.divisible_padding(height)
        self.padding = (0, 0, right, bottom)
        return sample

    def divisible_padding(self, size):
        return int(np.ceil(size / self.divisible) * self.divisible) - size


class RandomPad(Pad):
    def __init__(self, max_padding, fill=0):
        super().__init__(None, fill=fill)
        self.max_padding = self.size_from_number_or_iterable(max_padding)

    def __repr__(self):
        return self.__class__.__name__ + '(max_padding={0}, fill={1})'.format(self.max_padding, self.fill)

    def pre_transform(self, sample):
        sample = super().pre_transform(sample)

        expand_w = int(np.random.uniform(0, self.max_padding[0]))
        expand_h = int(np.random.uniform(0, self.max_padding[1]))
        left = int(np.random.uniform(0, expand_w))
        top = int(np.random.uniform(0, expand_h))
        right = expand_w - left
        bottom = expand_h - top
        self.padding = (left, top, right, bottom)
        return sample


class Crop(VisionTransform):
    def __init__(self, size, center=None, truncate_bbox=True, remove_bbox_outside=False):
        """
        :param size: output size
        :param center: center position of output in original image, `None` for center of original image
        :param truncate_bbox: truncate bbox to output size
        :param remove_bbox_outside: remove bbox which center is outside of output
        """
        self.size = self.size_from_number_or_iterable(size)

        if center is None:
            self.center = center
        else:
            self.center = self.size_from_number_or_iterable(center)

        self.truncate_bbox = truncate_bbox
        self.remove_bbox_outside = remove_bbox_outside

    def pre_transform(self, sample):
        image = sample['image']
        width, height = image.size
        if self.center is None:
            x, y = height // 2, width // 2
        else:
            x, y = self.center

        h, w = self.size
        h, w = h // 2, w // 2
        y1 = np.clip(y - h, 0, height)
        y2 = np.clip(y + h, 0, height)
        x1 = np.clip(x - w, 0, width)
        x2 = np.clip(x + w, 0, width)

        self.crop_rectangle = (x1, y1, x2, y2)
        self.image_size = image.size
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, center={1}, truncate_bbox={2}, remove_bbox_outside={3})'.format(
            self.size, self.center, self.truncate_bbox, self.remove_bbox_outside)

    def transform_image(self, image):
        return image.crop(self.crop_rectangle)

    def transform_bbox(self, bbox):
        x1, y1, x2, y2 = self.crop_rectangle
        w, h = self.image_size
        x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
        if self.remove_bbox_outside:
            # keep overlap with gt box IF center in sampled patch
            centers = (bbox[:, :2] + bbox[:, 2:]) / 2.0

            # mask in all gt boxes that above and to the left of centers
            m1 = (x1 < centers[:, 0]) * (y1 < centers[:, 1])

            # mask in all gt boxes that under and to the right of centers
            m2 = (x2 > centers[:, 0]) * (y2 > centers[:, 1])
        else:
            m1 = (x1 < (bbox[:, 2])) * (y1 < (bbox[:, 3]))
            m2 = (x2 > (bbox[:, 0])) * (y2 > (bbox[:, 1]))

        # mask in that both m1 and m2 are true
        mask = m1 * m2
        # take only matching gt boxes
        bbox = bbox[mask, :].copy()

        # adjust to crop (by subtracting crop's left,top)
        bbox[:] -= (x1, y1, x1, y1)

        # translate boxes
        if self.truncate_bbox:
            bbox[:, :2] = np.maximum(bbox[:, :2], 0)
            bbox[:, 2:] = np.minimum(bbox[:, 2:], 1)
        return bbox

    def transform_mask(self, mask):
        x1, y1, x2, y2 = self.crop_rectangle
        return mask[y1:y2, x1:x2]


class RandomCrop(Crop):
    def __init__(self, min_size, max_size, max_aspect_ratio=2, truncate_bbox=True, remove_bbox_outside=False):
        super().__init__(min_size, truncate_bbox=truncate_bbox, remove_bbox_outside=remove_bbox_outside)
        self.min_size = self.size_from_number_or_iterable(min_size)
        self.max_size = self.size_from_number_or_iterable(max_size)

        assert self.min_size[0] <= self.max_size[0]
        assert self.min_size[1] <= self.max_size[1]

        self.max_aspect_ratio = max_aspect_ratio

    def __repr__(self):
        return self.__class__.__name__ + '(min_size={0}, max_size={1}, max_aspect_ratio={2}, truncate_bbox={3}, remove_bbox_outside={4})'.format(self.min_size, self.max_size, self.max_aspect_ratio, self.truncate_bbox, self.remove_bbox_outside)

    def pre_transform(self, sample):
        image = sample['image']
        width, height = image.size

        max_width = min(width, self.max_size[0])
        max_height = min(height, self.max_size[1])

        w = np.random.uniform(self.min_size[0], max_width)
        h = np.random.uniform(self.min_size[1], max_height)

        # aspect ratio constraint
        if w / h > self.max_aspect_ratio:
            w = int(h * self.max_aspect_ratio)
        elif h / w > self.max_aspect_ratio:
            h = int(w * self.max_aspect_ratio)

        left = np.random.uniform(width - w)
        top = np.random.uniform(height - h)

        self.size = (int(w), int(h))
        self.center = (int(left + w // 2), int(top + h // 2))
        return super().pre_transform(sample)


class CutOut(VisionTransform):
    """Randomly mask out one patches from an image.
    """
    def __init__(self, max_size, fill=0):
        self.max_size = max_size
        self.fill = fill

    def __repr__(self):
        return self.__class__.__name__ + f'(max_size={self.max_size}, fill={self.fill})'

    def pre_transform(self, sample):
        image = sample['image']
        w, h = image.size
        y = np.random.randint(h)
        x = np.random.randint(w)
        size = int(np.random.uniform(0, self.max_size / 2))
        y1 = np.clip(y - size, 0, h)
        y2 = np.clip(y + size, 0, h)
        x1 = np.clip(x - size, 0, w)
        x2 = np.clip(x + size, 0, w)
        self.crop_rectangle = (x1, y1, x2, y2)
        self.image_size = image.size
        return sample

    def transform_image(self, image):
        draw = ImageDraw.Draw(image)
        draw.rectangle(self.crop_rectangle, fill=self.fill)
        return image

    def transform_bbox(self, bbox):
        x1, y1, x2, y2 = self.crop_rectangle
        w, h = self.image_size
        m1 = (x1 / w < (bbox[:, 0])) * (y1 / h < (bbox[:, 1]))
        m2 = (x2 / w > (bbox[:, 2])) * (y2 / h > (bbox[:, 3]))

        # mask that boxes are totally covered
        mask = m1 * m2

        bbox = np.delete(bbox, np.argwhere(mask), axis=0)
        return bbox

    def transform_mask(self, mask):
        mask = mask.copy()
        x1, y1, x2, y2 = self.crop_rectangle
        mask[y1:y2, x1:x2] = 0
        return mask


class HorizontalFlip(VisionTransform):
    def transform_image(self, image):
        return F.hflip(image)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox[:, 0::2] = 1 - bbox[:, -2::-2]
        return bbox

    def transform_mask(self, mask):
        return np.fliplr(mask)


class VerticalFlip(VisionTransform):
    def transform_image(self, image):
        return F.vflip(image)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox[:, 1::2] = 1 - bbox[:, -1::-2]
        return bbox

    def transform_mask(self, mask):
        return np.flipud(mask)


class Transpose(VisionTransform):
    def transform_image(self, image):
        return image.transpose(Image.TRANSPOSE)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox = bbox[:, [1, 0, 3, 2]]
        return bbox

    def transform_mask(self, mask):
        return mask.T


class ToRGB(VisionTransform):
    def transform_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image


class Grayscale(VisionTransform):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def transform_image(self, image):
        return F.to_grayscale(image, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)


class ColorJitter(VisionTransform, tvt.ColorJitter):
    def transform_image(self, image):
        return tvt.ColorJitter.__call__(self, image)

    def __repr__(self):
        return tvt.ColorJitter.__repr__(self)


class Posterize(VisionTransform):
    def __init__(self, bits):
        """
        Reduce the number of bits for each color channel.
        :param bits: The number of bits to keep for each channel (1-8).
        """
        self.bits = bits

    def __repr__(self):
        return self.__class__.__name__ + '(bits={0})'.format(self.bits)

    def transform_image(self, image):
        return ImageOps.posterize(image, self.bits)


class Solarize(VisionTransform):
    def __init__(self, threshold):
        """
        Invert all pixel values above a threshold.
        :param threshold: All pixels above this greyscale level are inverted.
        """
        self.threshold = threshold

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={0})'.format(self.threshold)

    def transform_image(self, image):
        return ImageOps.solarize(image, self.threshold)


class AutoContrast(VisionTransform):
    def transform_image(self, image):
        try:
            return ImageOps.autocontrast(image)
        except IOError as e:
            warnings.warn(str(e))
            return image


class Equalize(VisionTransform):
    def transform_image(self, image):
        return ImageOps.equalize(image)


class Invert(VisionTransform):
    def transform_image(self, image):
        return ImageOps.invert(image)


class CLAHE(VisionTransform):
    """Contrast Limited Adaptive Histogram Equalization"""
    def transform_image(self, image):
        return Image.fromarray((equalize_adapthist(np.asarray(image)) * 255).astype(np.uint8))


class SaltAndPepper(VisionTransform):
    def __init__(self, probability=0.001):
        self.probability = probability

    def __repr__(self):
        return self.__class__.__name__ + f'(probability={self.probability})'

    def transform_image(self, image):
        w, h = image.size
        noise = np.random.rand(h, w)
        probability = np.random.uniform(0, self.probability)
        threshold = 1 - probability

        salt = np.uint8(noise > threshold) * 255
        salt = Image.fromarray(salt)
        salt = salt.convert(image.mode)
        image = ImageChops.lighter(image, salt)

        pepper = np.uint8(noise > probability) * 255
        pepper = Image.fromarray(pepper)
        pepper = pepper.convert(image.mode)
        image = ImageChops.darker(image, pepper)

        return image


class ImageTransform(VisionTransform):
    enhance_ops = None

    def __init__(self, factor):
        self.factor = factor

    def __repr__(self):
        return self.__class__.__name__ + '(factor={0})'.format(self.factor)

    def transform_image(self, image):
        return self.enhance_ops(image).enhance(1 + self.factor)


class RandomImageTransform(ImageTransform):
    def __init__(self, factors=1.0, random='uniform'):
        super().__init__(1)
        if isinstance(factors, numbers.Number):
            assert factors >= 0
            factors = (-factors, factors)
        self.factors = factors
        self.random = random

    def __repr__(self):
        return self.__class__.__name__ + '(factors={0}, random={1})'.format(self.factors, self.random)

    def pre_transform(self, sample):
        self.factor = _random(self.random, self.factors)
        return super().pre_transform(sample)


class EnhanceColor(ImageTransform):
    enhance_ops = ImageEnhance.Color


class RandomAdjustColor(RandomImageTransform):
    enhance_ops = ImageEnhance.Color


class EnhanceContrast(ImageTransform):
    enhance_ops = ImageEnhance.Contrast


class RandomContrast(RandomImageTransform):
    enhance_ops = ImageEnhance.Contrast


class RandomSharpness(RandomImageTransform):
    enhance_ops = ImageEnhance.Sharpness


class RandomBrightness(RandomImageTransform):
    enhance_ops = ImageEnhance.Brightness


class JpegCompression(VisionTransform):
    def __init__(self, min_quality=45, max_quality=95):
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __repr__(self):
        return self.__class__.__name__ + f'(min_quality={self.min_quality}, max_quality={self.max_quality})'

    def transform_image(self, image):
        bytes_io = BytesIO()
        quality = int(np.random.uniform(self.min_quality, self.max_quality))
        image.save(bytes_io, 'JPEG', quality=quality)
        bytes_io.seek(0)
        return Image.open(bytes_io)


class ToTensor(VisionTransform):
    def __init__(self, normalize=TORCH_VISION_NORMALIZE):
        self.normalize = normalize

    def transform_image(self, image):
        image_t = F.to_tensor(image)
        if self.normalize:
            image_t = self.normalize(image_t)
        return image_t

    def transform_mask(self, mask):
        # numpy to tensor
        mask = np.ascontiguousarray(mask)
        return torch.from_numpy(mask)

    def post_transform(self, sample):
        sample['input'] = sample.pop('image')
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(normalize={self.normalize})'


class ToPILImage(VisionTransform):
    """undo `ToTensor`"""
    def __init__(self, denormalize=TORCH_VISION_DENORMALIZE):
        self.denormalize = denormalize

    def pre_transform(self, sample):
        sample = sample.copy()
        sample['image'] = sample.pop('input')
        return sample

    def transform_image(self, image):
        if self.denormalize:
            image = self.denormalize(image)
        return F.to_pil_image(image)

    def transform_mask(self, mask):
        return mask.numpy()

    def __repr__(self):
        return self.__class__.__name__ + f'(denormalize={self.denormalize})'


class ScaleBBox(VisionTransform):
    def __call__(self, sample):
        image = sample['image']
        boxes = sample['bbox']
        if boxes.size:
            boxes[:, 0::2] *= image.width
            boxes[:, 1::2] *= image.height
        return sample


class NormalizeBBox(VisionTransform):
    def __call__(self, sample):
        image = sample['image']
        boxes = sample['bbox']
        if boxes.size:
            boxes[:, 0::2] /= image.width
            boxes[:, 1::2] /= image.height

        return sample


class PerspectiveTransform(VisionTransform):
    def __init__(self, dst_corners, border='replicate'):
        self.dst_corners = dst_corners
        self.border = border

    def __repr__(self):
        return self.__class__.__name__ + f'(dst_corners={self.dst_corners}, border={self.border})'

    @staticmethod
    def get_perspective_transform(dst_corners, width, height):
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        box1 = np.asarray(dst_corners, dtype=np.float32) * np.array([width, height], dtype=np.float32)
        return cv2.getPerspectiveTransform(box0, box1)

    @staticmethod
    def warp_perspective(image_array, mat, border, interpolation):

        height, width = image_array.shape[:2]

        return cv2.warpPerspective(image_array, mat, (width, height), flags=interpolation,
                                   **cv2_border_mode_value(border))

    def pre_transform(self, sample):
        image = sample['image']
        self.perspective_matrix = self.get_perspective_transform(self.dst_corners,
                                                                 image.width, image.height)
        return super().pre_transform(sample)

    def transform_image(self, image):
        image_array = np.asarray(image)
        image_array = self.warp_perspective(image_array, self.perspective_matrix, self.border, cv2.INTER_LINEAR)
        return Image.fromarray(image_array)

    def transform_bbox(self, bbox):
        raise NotImplementedError()

    def transform_mask(self, mask):
        border = self.border if self.border == 'replicate' else None
        return self.warp_perspective(mask, self.perspective_matrix, border, cv2.INTER_NEAREST)


class Translate(PerspectiveTransform):
    def __init__(self, trans, border='replicate'):
        super().__init__(None, border)
        self.trans = trans

    def __repr__(self):
        return self.__class__.__name__ + f'(trans={self.trans}, border={self.border})'

    @property
    def trans(self):
        return self._trans

    @trans.setter
    def trans(self, t):
        assert isinstance(t, collections.Iterable)
        assert len(t) == 2
        self._trans = t
        self.dst_corners = [t, [1 + t[0], t[1]], [1 + t[0], t[1]], [t[0], 1 + t[1]]]


class RandomTranslate(Translate):
    def __init__(self, max_trans, border='replicate', random='uniform'):
        super().__init__((0, 0), border)
        if isinstance(max_trans, numbers.Number):
            assert max_trans > 0
            max_trans = (max_trans, max_trans)
        self.max_trans = max_trans
        self.random = random

    def __repr__(self):
        return self.__class__.__name__ + f'(max_trans={self.max_trans}, border={self.border}, random={self.random})'

    def pre_transform(self, sample):
        self.trans = (_random(self.random) * t for t in self.max_trans)
        return super().pre_transform(sample)


class HorizontalShear(PerspectiveTransform):
    def __init__(self, shear, border='replicate'):
        super().__init__(None, border)
        self.shear = shear

    def __repr__(self):
        return self.__class__.__name__ + f'(shear={self.shear}, border={self.border})'

    @property
    def shear(self):
        return self._shear

    @shear.setter
    def shear(self, dx):
        self._shear = dx
        self.dst_corners = [[dx, 0], [1 + dx, 0], [1 - dx, 1], [-dx, 1]]


class RandomHorizontalShear(HorizontalShear):
    def __init__(self, max_shear, border='replicate', random='uniform'):
        super().__init__(0, border)
        if isinstance(max_shear, numbers.Number):
            assert max_shear > 0
            max_shear = (-max_shear, max_shear)
        self.max_shear = max_shear
        self.random = random

    def __repr__(self):
        return self.__class__.__name__ + f'(max_shear={self.max_shear}, border={self.border}, random={self.random})'

    def pre_transform(self, sample):
        self.shear = _random(self.random, self.max_shear)
        return super().pre_transform(sample)


class VerticalShear(PerspectiveTransform):
    def __init__(self, shear, border='replicate'):
        super().__init__(None, border)
        self.shear = shear

    def __repr__(self):
        return self.__class__.__name__ + f'(shear={self.shear}, border={self.border})'

    @property
    def shear(self):
        return self._shear

    @shear.setter
    def shear(self, dy):
        self._shear = dy
        self.dst_corners = [[0, dy], [1, -dy], [1, 1 - dy], [0, 1 + dy]]


class RandomVerticalShear(VerticalShear):
    def __init__(self, max_shear, border='replicate', random='uniform'):
        super().__init__(0, border)
        if isinstance(max_shear, numbers.Number):
            assert max_shear > 0
            max_shear = (-max_shear, max_shear)
        self.max_shear = max_shear
        self.random = random

    def __repr__(self):
        return self.__class__.__name__ + f'(max_shear={self.max_shear}, border={self.border}, random={self.random})'

    def pre_transform(self, sample):
        self.shear = _random(self.random, self.max_shear)
        return super().pre_transform(sample)


class Skew(PerspectiveTransform):
    #        TopLeft, BottomLeft, BottomRight, TopRight
    direction_table = np.asarray([[[-1, 0], [0, 0], [0, 0], [0, 0]],
                                  [[0, -1], [0, 0], [0, 0], [0, 0]],
                                  [[0, -1], [0, 0], [0, 0], [0, +1]],
                                  [[0, 0], [0, 0], [0, 0], [0, +1]],
                                  [[0, 0], [0, 0], [0, 0], [-1, 0]],
                                  [[0, 0], [0, 0], [+1, 0], [-1, 0]],
                                  [[0, 0], [0, 0], [+1, 0], [0, 0]],
                                  [[0, 0], [0, 0], [0, +1], [0, 0]],
                                  [[0, 0], [0, -1], [0, +1], [0, 0]],
                                  [[0, 0], [0, -1], [0, 0], [0, 0]],
                                  [[0, 0], [+1, 0], [0, 0], [0, 0]],
                                  [[-1, 0], [+1, 0], [0, 0], [0, 0]]])

    def __init__(self, magnitude, direction=0, border='replicate'):
        """
        perspective skewing on images in one of 12 different direction.
        :param magnitude: percentage of image's size
        :param direction: skew direction as clock, e.g. 12 is skewing up, 6 is skewing down, 0 means random
        """
        super().__init__(None, border)
        self.magnitude = magnitude
        self.direction = direction

    def __repr__(self):
        return self.__class__.__name__ + f'(magnitude={self.magnitude}, direction={self.direction}, border={self.border})'

    def pre_transform(self, sample):
        direction = self.direction - 1 if self.direction != 0 else np.random.randint(0, len(self.direction_table))
        corners = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.dst_corners = corners + (self.direction_table[direction] * self.magnitude)
        return super().pre_transform(sample)


class RandomSkew(Skew):
    def __init__(self, max_magnitude, direction=0, border='replicate'):
        super().__init__(0, direction, border)
        self.max_magnitude = max_magnitude

    def __repr__(self):
        return self.__class__.__name__ + f'(max_magnitude={self.max_magnitude}, direction={self.direction}, border={self.border})'

    def pre_transform(self, sample):
        self.magnitude = np.random.uniform(-self.max_magnitude, self.max_magnitude)
        return super().pre_transform(sample)


class GridDistortion(VisionTransform):
    def __init__(self, num_steps=5, distort_limit=0.3, axis=None, border='replicate'):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.axis = axis
        self.border = border

    def __repr__(self):
        return self.__class__.__name__ + f'(num_steps={self.num_steps}, distort_limit={self.distort_limit}, axis={self.axis}, border={self.border})'

    def get_rand_steps(self, width):
        xsteps = 1 + np.random.uniform(-self.distort_limit, self.distort_limit, self.num_steps + 1)
        x_step = width // self.num_steps

        xx = np.zeros(width, np.float32)
        prev = 0
        for idx, x in enumerate(range(0, width, x_step)):
            start = x
            end = x + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step * xsteps[idx]

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur
        return xx

    def pre_transform(self, sample):
        width, height = sample['image'].size

        if self.axis is None or self.axis == 1:
            steps_x = self.get_rand_steps(width)
        else:
            steps_x = np.arange(width, dtype=np.float32)
        if self.axis is None or self.axis == 0:
            steps_y = self.get_rand_steps(height)
        else:
            steps_y = np.arange(height, dtype=np.float32)
        self.mesh_grid = np.meshgrid(steps_x, steps_y)

        return super().pre_transform(sample)

    def transform_image(self, image):
        image_array = np.asarray(image)
        image_array = cv2.remap(image_array, self.mesh_grid[0], self.mesh_grid[1],
                                interpolation=cv2.INTER_LINEAR, **cv2_border_mode_value(self.border))
        return Image.fromarray(image_array)

    def transform_bbox(self, bbox):
        raise NotImplementedError()

    def transform_mask(self, mask):
        return cv2.remap(mask, self.mesh_grid[0], self.mesh_grid[1],
                         interpolation=cv2.INTER_NEAREST, **cv2_border_mode_value(self.border))


class ElasticDeformation(VisionTransform):
    """Elastic deformation."""
    def __init__(self, alpha=1000, sigma=30, approximate=False):
        self.alpha = alpha
        self.sigma = sigma
        self.approximate = approximate

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, sigma={self.sigma}, approximate={self.approximate})'

    @staticmethod
    def elastic_indices(shape, alpha, sigma, approximate):
        """Elastic deformation of image as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.

        :see also:
            - https://github.com/albu/albumentations/blob/master/albumentations/augmentations/functional.py#L461
        """
        dx = (np.random.rand(*shape).astype(np.float32) * 2 - 1)
        dy = (np.random.rand(*shape).astype(np.float32) * 2 - 1)

        if approximate:
            # Approximate computation smooth displacement map with a large enough kernel.
            # On large images (512+) this is approximately 2X times faster
            ksize = sigma * 4 + 1
            cv2.GaussianBlur(dx, (ksize, ksize), sigma, dst=dx)
            cv2.GaussianBlur(dy, (ksize, ksize), sigma, dst=dy)
        else:
            dx = gaussian_filter(dx, sigma, mode="constant", cval=0)
            dy = gaussian_filter(dy, sigma, mode="constant", cval=0)

        dx *= alpha
        dy *= alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.asarray([x + dx,
                              y + dy])
        return indices

    @staticmethod
    def elastic_transform(image, indices, spline_order, mode='nearest'):
        if len(image.shape) == 3:
            result = np.empty_like(image)
            for i in range(image.shape[-1]):
                map_coordinates(image[..., i], indices, output=result[..., i], order=spline_order, mode=mode)
            return result
        return map_coordinates(image, indices, order=spline_order, mode=mode)

    def pre_transform(self, sample):
        image = sample['image']
        shape = (image.height, image.width)
        self.indices = self.elastic_indices(shape, self.alpha, self.sigma, self.approximate)
        return super().pre_transform(sample)

    def transform_image(self, image):
        image_array = np.asarray(image)
        image_array = self.elastic_transform(image_array, self.indices, 1)
        return Image.fromarray(image_array)

    def transform_bbox(self, bbox):
        raise NotImplementedError()

    def transform_mask(self, mask):
        return self.elastic_transform(mask, self.indices, 0)


class Rotate(VisionTransform):
    def __init__(self, angle, expand=False, fill=None):
        """
        :param angle: Rotation angle in degrees in counter-clockwise direction.
        :param expand: If true, expands the output to make it large enough to hold the entire rotated image.
        :param fill: color for area outside the rotated image.
        """
        self.angle = angle
        self.expand = expand
        self.fill = fill

    def __repr__(self):
        return self.__class__.__name__ + f'(angle={self.angle}, expand={self.expand}, fill={self.fill})'

    def transform_image(self, image):
        return image.rotate(angle=self.angle, expand=self.expand, fillcolor=self.fill)

    def transform_bbox(self, bbox):
        raise NotImplementedError()

    def transform_mask(self, mask):
        return skimage.transform.rotate(mask, self.angle, mode='constant', resize=self.expand,
                                        preserve_range=True, order=0).astype(mask.dtype)


class RandmonRotate(Rotate):
    def __init__(self, degrees, expand=False, fill=None):
        super().__init__(0, expand=expand, fill=fill)
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def __repr__(self):
        return self.__class__.__name__ + f'(degrees={self.degrees}, expand={self.expand}, fill={self.fill})'

    def pre_transform(self, sample):
        self.angle = np.random.uniform(self.degrees[0], self.degrees[1])
        return super().pre_transform(sample)


class TryApply(object):
    def __init__(self, transform: VisionTransform, max_trail=1):
        self.transform = transform
        self.max_trail = max_trail

    def __repr__(self):
        return self.__class__.__name__ + f'(transform={self.transform}, max_trail={self.max_trail})'

    def __call__(self, sample):
        n_bbox = len(sample.get('bbox', []))
        n_mask = {k: np.any(v) for k, v in sample.items() if k.startswith('mask')}

        for trail in range(self.max_trail):
            sample_t = self.transform(sample)

            # valid transform doens't remove all bbox or all mask
            n_bbox_n = len(sample.get('bbox', []))
            n_mask_n = {k: np.any(v) for k, v in sample.items() if k.startswith('mask')}
            assert n_bbox == n_bbox_n
            for k in n_mask:
                assert n_mask[k] == n_mask_n[k]

            bbox_ok = (n_bbox == 0 or len(sample_t.get('bbox', [])) > 0)
            if bbox_ok:
                mask_ok = True
                for k, v in n_mask.items():
                    if v and (not np.any(sample_t[k])):
                        mask_ok = False
                        break

                if mask_ok:
                    return sample_t
        return sample # fall back to no changes


class AutoAugment(RandomChoice):
    """AutoAugment - Learning Augmentation Policies from Data
    * https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html
    * https://github.com/DeepVoltaire/AutoAugment
    """

    @staticmethod
    def p_tran(p, t):
        if p <= 0:
            return []
        elif p >= 1:
            return [t]
        else:
            return [RandomApply(t)]

    @staticmethod
    def sub_trans(p0, p1):
        t = AutoAugment.p_tran(*p0) + AutoAugment.p_tran(*p1)
        if len(t) == 0:
            return PassThough
        elif len(t) == 1:
            return t[0]
        else:
            return Compose(t)


class ImageNetAugment(AutoAugment):
    def __init__(self, fill=(128, 128, 128)):
        linspace_0_1 = np.linspace(0.0, 0.9, 10)

        posterize = lambda i: Posterize(np.round(np.linspace(8, 4, 10), 0).astype(np.int)[i])
        solarize = lambda i: Solarize(np.linspace(256, 0, 10)[i])
        rotate = lambda i: Rotate(np.linspace(0, 30, 10)[i], fill=fill)
        color = lambda i: RandomAdjustColor(linspace_0_1[i], random='choice')
        sharpness = lambda i: RandomSharpness(linspace_0_1[i], random='choice')
        shearX = lambda i: RandomHorizontalShear(linspace_0_1[i] * 0.3, random='choice')
        contrast = lambda i: RandomContrast(linspace_0_1[i], random='choice')

        polices = (((0.4, posterize(8)), (0.6, rotate(9))),
                   ((0.6, solarize(5)),  (0.6, AutoContrast()),),
                   ((0.8, Equalize()),   (0.6, Equalize())),
                   ((0.6, posterize(7)), (0.6, posterize(6))),
                   ((0.4, Equalize()),   (0.2, solarize(4))),

                   ((0.4, Equalize()),   (0.8, rotate(8))),
                   ((0.6, solarize(3)),  (0.6, Equalize())),
                   ((0.8, posterize(5)), (1, Equalize())),
                   ((0.2, rotate(3)),    (0.6, solarize(8))),
                   ((0.6, Equalize()),   (0.4, posterize(6))),

                   ((0.8, rotate(8)),    (0,   color(0))),
                   ((0.4, rotate(9)),    (0.6, Equalize())),
                   ((0.0, Equalize()),   (0.8, Equalize())),
                   ((0.6, Invert()),     (1, Equalize())),
                   ((0.6, color(4)),     (1, contrast(8))),

                   ((0.8, rotate(8)),    (1,   color(2))),
                   ((0.8, color(8)),     (0.8, solarize(7))),
                   ((0.4, sharpness(7)), (0.6, Invert())),
                   ((0.6, shearX(5)),    (1, Equalize())),
                   ((0,   color(0)),     (0.6, Equalize())),

                   ((0.4, Equalize()),   (0.2, solarize(4))),
                   ((0.6, solarize(5)),  (0.6, AutoContrast())),
                   ((0.6, Invert()),     (1, Equalize())),
                   ((0.6, color(4)),     (1, contrast(8))),
                   ((0.8, Equalize()),   (0.6, Equalize()))
                   )

        trans = [AutoAugment.sub_trans(*p) for p in polices]
        super().__init__(trans)


class CIFAR10Augment(AutoAugment):
    def __init__(self, fill=(128, 128, 128)):
        linspace_0_1 = np.linspace(0.0, 0.9, 10)

        posterize = lambda i: Posterize(np.round(np.linspace(8, 4, 10), 0).astype(np.int)[i])
        solarize = lambda i: Solarize(np.linspace(256, 0, 10)[i])
        rotate = lambda i: Rotate(np.linspace(0, 30, 10)[i], fill=fill)
        color = lambda i: RandomAdjustColor(linspace_0_1[i], random='choice')
        sharpness = lambda i: RandomSharpness(linspace_0_1[i], random='choice')
        shearY = lambda i: RandomVerticalShear(linspace_0_1[i] * 0.3, random='choice')
        contrast = lambda i: RandomContrast(linspace_0_1[i], random='choice')
        translateX = lambda i: RandomTranslate((linspace_0_1[i] * 0.5, 0), random='choice')
        translateY = lambda i: RandomTranslate((0, linspace_0_1[i] * 0.5), random='choice')
        brightness = lambda i: RandomBrightness(linspace_0_1[i], random='choice')

        polices = (((0.1, Invert()), (0.2, contrast(6))),
                   ((0.7, rotate(2)),  (0.3, translateX(9)),),
                   ((0.8, sharpness(1)),   (0.9, sharpness(3))),
                   ((0.5, shearY(8)), (0.7, translateY(9))),
                   ((0.5, AutoContrast()),   (0.9, Equalize())),

                   ((0.2, shearY(7)),   (0.3, posterize(7))),
                   ((0.4, color(3)),  (0.6, brightness(7))),
                   ((0.3, sharpness(9)), (0.7, brightness(9))),
                   ((0.6, Equalize()),    (0.5, Equalize())),
                   ((0.6, contrast(7)),   (0.6, sharpness(5))),

                   ((0.7, color(7)),    (0,   translateX(0))),
                   ((0.3, Equalize()),    (0.4, AutoContrast())),
                   ((0.4, translateY(3)),   (0.2, sharpness(6))),
                   ((0.9, brightness(6)),     (0.2, color(8))),
                   ((0.5, solarize(2)),     (0, Invert())),

                   ((0.2, Equalize()),    (0.6, AutoContrast())),
                   ((0.2, Equalize()),     (0.8, Equalize())),
                   ((0.9, color(9)),        (0.6, Equalize())),
                   ((0.8, AutoContrast()),    (0.2, solarize(8))),
                   ((0.1, brightness(3)),     (0,  color(0))),

                   ((0.4, solarize(5)),   (0.9, AutoContrast())),
                   ((0.9, translateY(9)),  (0.7, translateY(9))),
                   ((0.9, AutoContrast()),     (0.8, solarize(3))),
                   ((0.8, Equalize()),     (0.1, Invert())),
                   ((0.7, translateY(9)),     (0.9, AutoContrast()))
                   )

        trans = [AutoAugment.sub_trans(*p) for p in polices]
        super().__init__(trans)


class SVHNAugment(AutoAugment):
    def __init__(self, fill=(128, 128, 128)):
        linspace_0_1 = np.linspace(0.0, 0.9, 10)

        solarize = lambda i: Solarize(np.linspace(256, 0, 10)[i])
        rotate = lambda i: Rotate(np.linspace(0, 30, 10)[i], fill=fill)
        shearX = lambda i: RandomHorizontalShear(linspace_0_1[i] * 0.3, random='choice')
        shearY = lambda i: RandomVerticalShear(linspace_0_1[i] * 0.3, random='choice')
        contrast = lambda i: RandomContrast(linspace_0_1[i], random='choice')
        translateX = lambda i: RandomTranslate((linspace_0_1[i] * 0.5, 0), random='choice')
        translateY = lambda i: RandomTranslate((0, linspace_0_1[i] * 0.5), random='choice')

        polices = (((0.9, shearX(4)), (0.2, Invert())),
                   ((0.9, shearY(8)),  (0.7, Invert()),),
                   ((0.6, Equalize()),   (0.6, solarize(6))),
                   ((0.9, Invert()), (0.6, Equalize())),
                   ((0.6, Equalize()),   (0.9, rotate(3))),

                   ((0.9, shearX(4)),   (0.8, AutoContrast())),
                   ((0.9, shearY(8)),  (0.4, Invert())),
                   ((0.9, shearY(5)), (0.2, solarize(6))),
                   ((0.9, Invert()),    (0.8, AutoContrast())),
                   ((0.6, Equalize()),   (0.9, rotate(3))),

                   ((0.9, shearX(4)),    (0.3,   solarize(3))),
                   ((0.8, shearY(8)),    (0.7, Invert())),
                   ((0.9, Equalize()),   (0.6, translateY(6))),
                   ((0.9, Invert()),     (0.6, Equalize())),
                   ((0.3, contrast(3)),     (0.8, rotate(4))),

                   ((0.8, Invert()),    (0, translateY(2))),
                   ((0.7, shearY(6)),     (0.4, solarize(8))),
                   ((0.6, Invert()),        (0.8, rotate(4))),
                   ((0.3, shearY(7)),    (0.9, translateX(3))),
                   ((0.1, shearX(6)),     (0.6,  Invert())),

                   ((0.7, solarize(2)),   (0.6, translateY(7))),
                   ((0.8, shearY(4)),  (0.8, Invert())),
                   ((0.7, shearX(9)),     (0.8, translateY(3))),
                   ((0.8, shearY(5)),     (0.7, AutoContrast())),
                   ((0.7, shearX(2)),     (0.1, Invert()))
                   )

        trans = [AutoAugment.sub_trans(*p) for p in polices]
        super().__init__(trans)