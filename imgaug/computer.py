from io import BytesIO
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

    def __init__(self, filename, stream=None, action=lambda *args: args[0], label=None, labeller=image.Labeller(), normalizer=image.Normalizer(size=256, canvas=True), augmenter=image.Augmenter()):
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
