from os import path
from logging import debug, info
from struct import unpack
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from imgaug import filters


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

    SIZE = 64
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
    transformations by applying a list of predefined filters.
    Filter is expected to be a callable object accepting the image data and a
    value, that falls within the acceptable VALUES range.

    Examples
    --------
    >>> aug = Augmenter(cutoff=.5)
    >>> aug('resources/bag.png').__class__.__name__
    'generator'
    '''

    CUTOFF = 1.
    FILTERS = (filters.Blur(), filters.Flip(), filters.Gamma(), filters.Gaussian(), filters.Noise(), filters.Rescale(), filters.Rotate(), filters.Shift('*'), filters.Shift('h'), filters.Shift('v'), filters.Skew(), filters.Pixel('max'), filters.Pixel('median'), filters.Pixel('min'), filters.Pixel('mode'), filters.Unsharp())

    def __init__(self, cutoff=CUTOFF):
        self.cutoff = float(cutoff) or self.CUTOFF
    
    def __call__(self, name):
        info('apply transformations to image')
        img = self._img(name)
        yield img
        for _filter in self.FILTERS:
            for val in self._cut(_filter.VALUES):
                filtered = _filter(img, val)
                if filtered is not None:
                    debug('applied filter %s with value %s', _filter.__class__.__name__, val)
                    yield(filtered)

    def _img(self, name):
        if isinstance(name, np.ndarray):
            return name
        return plt.imread(name)

    def _cut(self, rng):
        if self.cutoff >= 1 or isinstance(rng, tuple):
            return rng
        sl = round(len(rng) * self.cutoff) or 1
        return rng[:sl]
