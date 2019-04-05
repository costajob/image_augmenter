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
        name, _ = path.basename(name).split('.')
        for sep in self.separators:
            label = self._tokenize(name, sep)
            if label:
                return label
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

    @property
    def is_bkg(self):
        filepath = isinstance(self.canvas, str) and path.isfile(self.canvas)
        stream = hasattr(self.canvas, 'read')
        return filepath or stream

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
        if self.is_bkg:
            info('applying background')
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
    - shifting 
    - skewing

    Warning
    -------
    The transformers methods are collected bu iterating on attributes starting with 
    the prefix '_tr': be aware of that when extending this class.

    Examples
    --------
    >>> aug = Augmenter(cutoff=.5)
    >>> aug('resources/bag.png').__class__.__name__
    'generator'
    '''

    CUTOFF = 1.
    RESCALE_MODE = 'constant'
    NOISE_MODE = 'speckle'
    FLIP = (np.s_[:, ::-1], np.s_[::-1, :])
    BLUR = np.arange(2, 12, 1)
    GAMMA = np.arange(.1, 2.6, .05)
    NOISE = np.arange(.0005, .0300, .0005)
    SCALE = np.arange(1.05, 2.3, .01)
    ROTATE = np.arange(-155, 155, 0.7)
    SHIFT = np.arange(3, 300, 3)
    SKEW = np.arange(-.8, .8, .13)
    RANGES = (BLUR, FLIP, GAMMA, NOISE, SCALE, ROTATE, SHIFT, SHIFT, SHIFT, SKEW)

    def __init__(self, cutoff=CUTOFF):
        self.cutoff = float(cutoff) or self.CUTOFF
        self.ranges = [self._cut(rng) for rng in self.RANGES]
        self.count = self._count()
    
    @property
    def transformers(self):
        names = sorted(tr for tr in dir(self) if tr.startswith('_tr'))
        return [getattr(self, name) for name in names]

    def __call__(self, name):
        img = self._img(name)
        yield img
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
        if self.cutoff >= 1 or isinstance(rng, tuple):
            return rng
        else:
            _min = rng.min()
            start = _min if _min > 0 else _min * self.cutoff
            stop = rng.max() * self.cutoff
            step = f'{rng[1]-rng[0]:.4f}'
            return np.arange(start, stop, float(step))

    def _count(self):
        return sum(len(r) for r in self.ranges) + 1
    
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

    def _tr_shift(self, img, v):
        h, w, _ = img.shape
        if v < w and v < h:
            yield self._shift(img, (v, v))

    def _tr_shift_h(self, img, x):
        _, w, _ = img.shape
        if x < w:
            yield self._shift(img, (x, 0))

    def _tr_shift_v(self, img, y):
        h, _, _ = img.shape
        if y < h:
            yield self._shift(img, (0, y))

    def _tr_skew(self, img, shear):
        tf = AffineTransform(shear=shear)
        yield warp(img, inverse_map=tf)

    def _shift(self, img, vector):
        tf = AffineTransform(translation=vector)
        return warp(img, inverse_map=tf, mode='wrap')

    def _RGB(self, img):
        return img.shape[-1] == 3
