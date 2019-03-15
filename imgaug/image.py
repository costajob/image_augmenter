from math import floor
from os import path
from struct import unpack
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.exposure import adjust_gamma
from skimage.transform import rescale, rotate
from skimage.util import random_noise
import numpy as np
from imgaug.logger import BASE as logger


class Normalizer:
    '''
    Synopsis
    --------
    Normalizes the specified image by:
    - resizing the largest dimension to specified max size
    - creating a squared canvas by max size and pasting the image in front of it

    Examples
    --------
    >>> norm = Normalizer(size=128, canvas=True)
    >>> img = norm('resources/bag.png')
    >>> img.size
    (128, 128)
    '''

    SIZE = 32
    TRANSPARENT = (255, 0, 0, 0)
    WHITE = (255, 255, 255)
    CANVAS = False
    RGBA = 'RGBA'
    PNG = 'PNG'
    
    def __init__(self, size=SIZE, canvas=CANVAS):
        self.size = int(size)
        self.canvas = canvas
        self.bkg = path.isfile(str(canvas))

    def __call__(self, name):
        img = self._resize(name)
        if img:
            if self.canvas:
                img = self._canvas(img)
            return img

    def _resize(self, name):
        img = self._img(name)
        if img.format == self.PNG:
            img = img.convert(self.RGBA)
        w, h = img.size
        _max = max(w, h)
        ratio = _max / self.size
        size = (int(w // ratio), int(h // ratio))
        logger.info('resizing image to %r', size)
        return img.resize(size)

    def _img(self, name):
        if hasattr(name, 'size'):
            return name
        return Image.open(name)

    def _canvas(self, img):
        size = (self.size, self.size)
        offset = self._offset(img)
        if self.bkg:
            logger.info('applying background %s', path.basename(self.canvas))
            c = Image.open(self.canvas).convert(img.mode)
            c = c.resize(size)
            c.paste(img, offset, img.convert(self.RGBA))
        else:
            logger.info('applying squared canvas %r', size)
            c = Image.new(img.mode, size, self._color(img))
            c.paste(img, offset)
        return c
    
    def _color(self, img):
        if img.mode == self.RGBA:
            return self.TRANSPARENT
        if self.canvas is True:
            return self.WHITE
        return unpack('BBB', bytes.fromhex(self.canvas))

    def _offset(self, img):
        w, h = img.size
        return ((self.size - w) // 2, (self.size - h) // 2)


class Augmenter:
    '''
    Synopsis
    --------
    Performs data augmentation on the specified Numpy image by applying a set of 
    transformations:
    - blurring
    - flipping
    - adjusting gamma
    - rescaling and cropping
    - adding random noise
    - rotating

    Warning
    -------
    The transformers methods are collected bu iterating on attributes starting with 
    the prefix '_tr': be aware of that when extending this class.

    Examples
    --------
    >>> aug = Augmenter(.75)
    >>> aug.count
    769
    '''

    CUTOFF = 1.
    RESCALE_MODE = 'constant'
    NOISE_MODE = 'speckle'
    BLUR = range(2, 7, 1)
    FLIP = (np.s_[:, ::-1], np.s_[::-1, :])
    GAMMA = np.arange(.1, 3., .02)
    NOISE = np.arange(.0005, .0260, .0003)
    SCALE = np.arange(1.05, 3.35, .0075)
    ROTATE = np.arange(-157, 157, .65)
    RANGES = (BLUR, FLIP, GAMMA, NOISE, SCALE, ROTATE)

    def __init__(self, cutoff=CUTOFF):
        self.cutoff = float(cutoff)
        self.ranges = [self._cut(rng) for rng in self.RANGES]
        self.count = self._count()
    
    @property
    def transformers(self):
        names = sorted(tr for tr in dir(self) if tr.startswith('_tr'))
        return [getattr(self, name) for name in names]

    def __call__(self, name):
        img = self._img(name)
        yield img
        if self.cutoff:
            logger.info('applying a set of %d transformations', self.count)
            for rng, tr in zip(self.ranges, self.transformers):
                for val in rng:
                    yield from tr(img, val)

    def _img(self, name):
        if isinstance(name, np.ndarray):
            return name
        return plt.imread(name)

    def _cut(self, rng):
        if self.cutoff >= 1:
            return rng
        cut = floor(len(rng) * self.cutoff) or 1
        return rng[:cut]

    def _count(self):
        count = 1
        if self.cutoff:
            count = sum(len(r) for r in self.ranges) + 1
        return count
    
    def _tr_blur(self, img, axe):
        yield uniform_filter(img, size=(axe, axe, 1))

    def _tr_flip(self, img, sl):
        yield img[sl]

    def _tr_gamma(self, img, gm):
        yield adjust_gamma(img, gamma=gm, gain=.9)

    def _tr_noise(self, img, var):
        yield random_noise(img, mode=self.NOISE_MODE, var=var)

    def _tr_rescale(self, img, sc):
        _data = rescale(img, sc, mode=self.RESCALE_MODE, anti_aliasing=True, multichannel=True)
        h, w, _ = _data.shape
        y, x, _ = img.shape
        cx = w // 2 - (x // 2)
        cy = h // 2 - (y // 2)
        yield _data[cy:cy+y, cx:cx+x, :]

    def _tr_rotate(self, img, ang):
        cval = 1. if self._RGB(img) else 0
        yield rotate(img, ang, cval=cval)

    def _RGB(self, img):
        return img.shape[-1] == 3
