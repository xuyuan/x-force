import numpy as np


def rle_encoding(x, label=1):
    """
    The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

    Parameters
    ----------
    x: numpy array of shape (height, width)
    lable: (int)
    Returns: run length as list
    """
    pixels = (x.T.flatten() == label).astype(np.uint8)  # .T sets Fortran order down-then-right
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs


# https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if isinstance(mask_rle, str):
        mask_rle = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).transpose()


if __name__ == '__main__':
    mask = np.uint8(np.random.random((100, 100)) > 0.5)
    rle = rle_encoding(mask)
    rle_str = " ".join([str(i) for i in rle])
    mask2 = rle_decode(rle_str, mask.shape)
    assert np.allclose(mask, mask2)