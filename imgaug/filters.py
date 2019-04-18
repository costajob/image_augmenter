import numpy as np
from PIL import Image
from PIL.ImageFilter import MaxFilter, MedianFilter, MinFilter, ModeFilter, UnsharpMask
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.exposure import adjust_gamma
from skimage.transform import AffineTransform, rescale, rotate, warp
from skimage.util import random_noise


class Blur:
    VALUES = range(2, 10, 1)

    def __call__(self, img, axe):
        return uniform_filter(img, size=(axe, axe, 1))


class Flip:
    VALUES = (np.s_[:, ::-1], np.s_[::-1, :])

    def __call__(self, img, sl):
        return img[sl]


class Gamma:
    VALUES = np.arange(.1, 2.55, .05)
    GAIN = .9

    def __call__(self, img, gm):
        return adjust_gamma(img, gamma=gm, gain=self.GAIN)


class Gaussian:
    VALUES = np.arange(.2, 1.5, .1)

    def __call__(self, img, sigma):
        return gaussian_filter(img, sigma)


class Noise:
    VALUES = np.arange(.001, .0301, .001)
    MODE = 'speckle'

    def __call__(self, img, var):
        return random_noise(img, mode=self.MODE, var=var)


class Rescale:
    VALUES = np.arange(1.05, 2.35, .05)
    MODE = 'constant'

    def __call__(self, img, sc):
        data = rescale(img, sc, mode=self.MODE, anti_aliasing=True, multichannel=True)
        h, w, _ = data.shape
        y, x, _ = img.shape
        cx = w // 2 - (x // 2)
        cy = h // 2 - (y // 2)
        return data[cy:cy+y, cx:cx+x, :]


class Rotate:
    VALUES = range(-155, 156, 2)

    def __call__(self, img, ang):
        cval = 1. if self._RGB(img) else 0
        return rotate(img, ang, cval=cval)

    def _RGB(self, img):
        return img.shape[-1] == 3


class Shift:
    VALUES = range(1, 512, 1)
    VERTICAL = 'v'
    HORIZONTAL = 'h'

    def __init__(self, mode='*'):
        self.vertical = mode == self.VERTICAL
        self.horizontal = mode == self.HORIZONTAL

    def __call__(self, img, v):
        if self._valid(img, v):
            vector = self._vector(v)
            tf = AffineTransform(translation=vector)
            return warp(img, inverse_map=tf, mode='wrap')

    def _vector(self, v):
        if self.vertical:
            return (0, v)
        elif self.horizontal:
            return (v, 0)
        return (v, v)

    def _valid(self, img, v):
        h, w, _ = img.shape
        if self.vertical:
            return v < h
        elif self.horizontal:
            return v < w
        return v < w and v < h


class Skew:
    VALUES = np.arange(-.8, .9, .13)

    def __call__(self, img, shear):
        tf = AffineTransform(shear=shear)
        return warp(img, inverse_map=tf)


class Pixel:
    VALUES = range(1, 14, 2)
    FILTERS = {'max': MaxFilter, 'median': MedianFilter, 'min': MinFilter, 'mode': ModeFilter}

    def __init__(self, _filter):
        self.filter = self.FILTERS.get(_filter, MinFilter)

    def __call__(self, img, size):
        img = Image.fromarray(img)
        filtered = img.filter(self.filter(size))
        return np.array(filtered)


class Unsharp:
    VALUES = range(1, 51, 1)

    def __call__(self, img, radius):
        img = Image.fromarray(img)
        filtered = img.filter(UnsharpMask(radius))
        return np.array(filtered)
