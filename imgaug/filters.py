import numpy as np
from PIL import Image
from PIL.ImageFilter import MaxFilter, MedianFilter, MinFilter, ModeFilter, UnsharpMask
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.exposure import adjust_gamma
from skimage.transform import AffineTransform, rescale, rotate, warp
from skimage.util import random_noise


class Blur:
    VALUES = range(2, 8, 1)

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
    VALUES = np.arange(1.05, 2.05, .03)
    MODE = 'constant'

    def __call__(self, img, sc):
        data = rescale(img, sc, mode=self.MODE, anti_aliasing=True, multichannel=True)
        h, w, _ = data.shape
        y, x, _ = img.shape
        cx = w // 2 - (x // 2)
        cy = h // 2 - (y // 2)
        return data[cy:cy+y, cx:cx+x, :]


class Rotate:
    VALUES = range(-155, 156, 1)
    MODE = 'edge'

    def __call__(self, img, ang):
        if ang:
            cval = 1. if self._RGB(img) else 0
            return rotate(img, ang, cval=cval, mode=self.MODE)

    def _RGB(self, img):
        return img.shape[-1] == 3


class Shift:
    VALUES = range(-512, 512, 1)
    MODE = 'edge'
    RATIO = 3
    VERTICAL = 'v'
    HORIZONTAL = 'h'

    def __init__(self, mode='*'):
        self.vertical = mode == self.VERTICAL
        self.horizontal = mode == self.HORIZONTAL

    def __call__(self, img, vec):
        if self._valid(img, vec):
            vector = self._vector(vec)
            tf = AffineTransform(translation=vector)
            return warp(img, inverse_map=tf, mode=self.MODE)

    def _vector(self, vec):
        if self.vertical:
            return (0, vec)
        elif self.horizontal:
            return (vec, 0)
        return (vec, vec)

    def _valid(self, img, vec):
        if vec:
            vec = abs(vec)
            h, w, _ = [d // self.RATIO for d in img.shape]
            if self.vertical:
                return vec < h
            elif self.horizontal:
                return vec < w
            return vec < min(w, h)


class Skew:
    VALUES = np.arange(-.3, .4, .05)
    MODE = 'edge'
    MIN = .09

    def __call__(self, img, shear):
        if abs(shear) > self.MIN:
            tf = AffineTransform(shear=shear)
            return warp(img, inverse_map=tf, mode=self.MODE)


class Pixel:
    VALUES = range(3, 12, 2)
    FILTERS = {'max': MaxFilter, 'median': MedianFilter, 'min': MinFilter, 'mode': ModeFilter}
    OVERSIZES = {300: 7, 200: 5, 100: 3}

    def __init__(self, _filter):
        self.filter = self.FILTERS.get(_filter, MinFilter)

    def __call__(self, img, size):
        if self._oversized(img, size):
            return
        img = Image.fromarray(img)
        filtered = img.filter(self.filter(size))
        return np.array(filtered)

    def _oversized(self, img, size):
        h, w, _ = img.shape
        dim = min(h, w)
        for d, s in self.OVERSIZES.items():
            if dim < d and size > s:
                return True


class Unsharp:
    VALUES = range(1, 51, 1)

    def __call__(self, img, radius):
        img = Image.fromarray(img)
        filtered = img.filter(UnsharpMask(radius))
        return np.array(filtered)
