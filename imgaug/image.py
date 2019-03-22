from math import floor
from os import path
from logging import debug, info
from struct import unpack
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.exposure import adjust_gamma
from skimage.transform import AffineTransform, rescale, rotate, warp
from skimage.util import random_noise
import numpy as np


class Labeller:
    '''
    Synopsis
    --------
    Identifies the label based on the specified name, by checking special chars and
    specified meaningful digits length.

    Examples
    --------
    >>> lbl = Labeller(digits=10)
    >>> lbl('resources/109-602-3906-001-c-suit-veletta-albino.jpg')
    '1096023906'
    '''

    MEANINGFUL_DIGITS = 13
    SEPARATORS = ('-', '_')

    def __init__(self, digits=MEANINGFUL_DIGITS, separators=SEPARATORS):
        self.digits = int(digits)
        self.separators = separators

    def __call__(self, name):
        name = path.basename(name)
        for sep in self.separators:
            label = self._tokenize(name, sep)
            if label:
                return label
        return self._plain(name)

    def _plain(self, name):
        name, _ = name.split('.')
        return name[:self.digits]

    def _tokenize(self, name, sep):
        if name.count(sep) > 0:
            label = ''
            for token in name.split(sep):
                label += token
                if len(label) >= self.digits:
                    return label 


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
    >>> img.shape
    (128, 128, 4)
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
            return np.array(img)

    def _resize(self, name):
        img = Image.open(name)
        if img.format == self.PNG:
            img = img.convert(self.RGBA)
        w, h = img.size
        _max = max(w, h)
        ratio = _max / self.size
        size = (int(w // ratio), int(h // ratio))
        info('resizing image to %r', size)
        return img.resize(size)

    def _canvas(self, img):
        size = (self.size, self.size)
        offset = self._offset(img)
        if self.bkg:
            info('applying background %s', path.basename(self.canvas))
            c = Image.open(self.canvas).convert(img.mode)
            c = c.resize(size)
            c.paste(img, offset, img.convert(self.RGBA))
        else:
            info('applying squared canvas %r', size)
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
    - skewing

    Warning
    -------
    The transformers methods are collected bu iterating on attributes starting with 
    the prefix '_tr': be aware of that when extending this class.

    Examples
    --------
    >>> aug = Augmenter(cutoff=.5)
    >>> aug.count
    498
    '''

    CUTOFF = 1.
    RESCALE_MODE = 'constant'
    NOISE_MODE = 'speckle'
    BLUR = range(2, 7, 1)
    FLIP = (np.s_[:, ::-1], np.s_[::-1, :])
    GAMMA = np.arange(.1, 3.2, .02)
    NOISE = np.arange(.0005, .025, .0005)
    SCALE = np.arange(1.05, 3.6, .01)
    ROTATE = np.arange(-155, 155, .6)
    SKEW = np.arange(-.8, .8, .1)
    RANGES = (BLUR, FLIP, GAMMA, NOISE, SCALE, ROTATE, SKEW)

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
            info('applying a set of %d transformations', self.count)
            for rng, tr in zip(self.ranges, self.transformers):
                for val in rng:
                    debug('applying %s(%r)', tr.__name__, val)
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

    def _tr_skew(self, img, shear):
        tf = AffineTransform(shear=shear)
        yield warp(img, inverse_map=tf)

    def _RGB(self, img):
        return img.shape[-1] == 3
