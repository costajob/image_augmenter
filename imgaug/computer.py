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
    Creates a set of compressed files by: 
    * iterating images within the specified folder
    * recognizing each label
    * creating an archive with label name
    * normalizing and augmenting the image
    * archive images within the specified label-named folder
    * generating a distinct compressed file by checking the x-zip attribute

    Examples
    --------
    >>> zipper = Zipper('resources')
    >>> len(list(zipper.files))
    3
    '''

    MAXLEN = 13
    SIZE = 64
    EXTS = {'png', 'jpg', 'jpeg'}
    X_ZIP = 15000

    def __init__(self, folder, size=SIZE, x_zip=X_ZIP, cutoff=1., labeller=image.Labeller(), normalizer_cls=image.Normalizer, augmenter_cls=image.Augmenter):
        self.files = self._files(folder)
        self.labeller = labeller
        self.norm = normalizer_cls(size)
        self.augmenter = augmenter_cls(cutoff)
        self.x_zip = int(x_zip)
        self.zipname = f'dataset_{self.timestamp}'
    
    @property
    def timestamp(self):
        return str(datetime.utcnow().timestamp()).replace('.', '')

    def __call__(self):
        for i, accumulator in enumerate(self):
            zipname = f'{self.zipname}{i:02}.zip'
            info('creating compressed file %s', zipname)
            with ZipFile(zipname, 'w') as zfile:
                for filepath, archive in accumulator:
                    zfile.write(filepath, arcname=archive)

    def __iter__(self):
        tmpdir = mkdtemp(prefix='images')
        count = 0
        accumulator = []
        for filepath in self.files:
            info('processing file %s', path.basename(filepath))
            basename = self._basename()
            label = self.labeller(filepath)
            norm = self.norm(filepath)
            ext = self._ext(filepath)
            for i, data in enumerate(self.augmenter(norm)):
                count += 1
                name = f'{basename}{i:03}.{ext}'
                tmpname = path.join(tmpdir, name)
                plt.imsave(tmpname, data)
                archive = path.join(label, name)
                accumulator.append((tmpname, archive))
                if count % self.x_zip == 0:
                    yield(accumulator)
                    accumulator = []
        if accumulator:
            yield(accumulator)

    def _files(self, folder):
        folder = path.expanduser(folder)
        return (f for f in glob(path.join(folder, '*')) if self._valid(f))

    def _basename(self):
        return ''.join(choices(ascii_letters + digits, k=self.MAXLEN))

    def _valid(self, filepath):
        ext = self._ext(filepath)
        return ext in self.EXTS

    def _ext(self, filepath):
        return filepath.rsplit('.', 1)[-1].lower()
