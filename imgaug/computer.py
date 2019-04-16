from datetime import datetime
from glob import glob
from os import path
from io import BytesIO
from logging import info
from random import choices
from string import ascii_letters, digits
from tempfile import mkdtemp
from zipfile import ZipFile
from matplotlib import pyplot as plt
from imgaug import image


class Persister:
    '''
    Synopsis
    --------
    Accepts an image path as an input (a stream optionally), normalise it and 
    augment it by using the specified collaborators.
    Transforms each of the augmented image to stream-like object and calls the
    specified action function.

    Examples
    --------
    >>> pers = Persister('resources/bag.png')
    >>> pers.ext
    'png'
    '''

    JPG = 'jpg'
    PNG = 'png'

    def __init__(self, filename, stream=None, action=lambda *args: args[0], label=None, labeller=image.Labeller(), normalizer=image.Normalizer(), augmenter=image.Augmenter()):
        self.action = action
        self.label = label or labeller(filename)
        self.norm = normalizer(stream or filename)
        self.ext = self._ext()
        self.augmenter = augmenter

    def __iter__(self):
        for i, data in enumerate(self.augmenter(self.norm)):
            name = f'{self.label}_{i:03}.{self.ext}'
            stream = BytesIO()
            plt.imsave(stream, data)
            filepath = self.action(name, stream)
            yield(self.label, filepath)

    def _ext(self):
        _, _, c = self.norm.shape
        return self.PNG if c == 4 else self.JPG


class Zipper:
    '''
    Synopsis
    --------
    Creates a compressed file by iterating files within the specified folder, recognizing
    each label, creating an archive with label name, normalizing and augmenting the file
    and putting them into the specified label-named archive.

    Examples
    --------
    >>> zipper = Zipper('resources')
    >>> len(list(zipper.files))
    3
    '''

    MAXLEN = 16
    EXTS = {'png', 'jpg', 'jpeg'}

    def __init__(self, folder, labeller=image.Labeller(), normalizer=image.Normalizer(), augmenter=image.Augmenter()):
        self.files = self._files(folder)
        self.labeller = labeller
        self.norm = normalizer
        self.augmenter = augmenter
        self.zipname = f'dataset_{self.timestamp}.zip'
    
    @property
    def timestamp(self):
        return str(datetime.utcnow().timestamp()).replace('.', '')

    def __call__(self):
        info('creating compressed file %s', self.zipname)
        with ZipFile(self.zipname, 'w') as zfile:
            for filepath, archive in self:
                zfile.write(filepath, arcname=archive)

    def __iter__(self):
        tmpdir = mkdtemp(prefix='images')
        for filepath in self.files:
            label = self.labeller(filepath)
            norm = self.norm(filepath)
            ext = self._ext(filepath)
            for data in self.augmenter(norm):
                name = self._filename(ext)
                tmpname = path.join(tmpdir, name)
                plt.imsave(tmpname, data)
                archive = path.join(label, name)
                yield(tmpname, archive)

    def _files(self, folder):
        folder = path.expanduser(folder)
        return (f for f in glob(path.join(folder, '*')) if self._valid(f))

    def _filename(self, ext):
        letters = ''.join(choices(ascii_letters + digits, k=self.MAXLEN))
        return f'{letters}.{ext}'

    def _valid(self, filepath):
        ext = self._ext(filepath)
        return ext in self.EXTS

    def _ext(self, filepath):
        return filepath.rsplit('.', 1)[-1].lower()
