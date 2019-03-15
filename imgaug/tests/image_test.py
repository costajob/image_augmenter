import unittest
from PIL import Image
from imgaug import image


class TestTraining(unittest.TestCase):
    def test_normalization_path(self):
        norm = image.Normalizer(size=64)
        img = norm('resources/bag.png')
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 42)
        self.assertEqual(img.mode, 'RGBA')

    def test_normalization_pil_img(self):
        norm = image.Normalizer(size=64)
        img = norm(Image.open('resources/bag.png'))
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 42)
        self.assertEqual(img.mode, 'RGBA')

    def test_normalization_canvas(self):
        norm = image.Normalizer(size=64, canvas=True)
        img = norm('resources/bag.png')
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGBA')

    def test_normalization_colored_canvas(self):
        norm = image.Normalizer(size=64, canvas='FF0000')
        img = norm('resources/bag.png')
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGBA')

    def test_normalization_bkg_canvas(self):
        norm = image.Normalizer(size=64, canvas=f'resources/office.png')
        img = norm('resources/bag.png')
        w, h = img.size
        self.assertEqual(w, 64)
        self.assertEqual(h, 64)
        self.assertEqual(img.mode, 'RGBA')
   
    def test_augmenting_attributes(self):
        aug = image.Augmenter()
        self.assertEqual(len(aug.transformers), 6)
        self.assertEqual(aug.count, 1029)
    
    def test_augmenting(self):
        aug = image.Augmenter(.01)
        images = list(aug('resources/bag.png'))
        self.assertEqual(len(images), aug.count)
        for img in images:
            h, w, c = img.shape
            self.assertEqual(h, 200)
            self.assertEqual(w, 300)
            self.assertEqual(c, 3)

    def test_augmenting_skip(self):
        aug = image.Augmenter(0)
        images = list(aug('resources/bag.png'))
        self.assertEqual(aug.count, 1)
        self.assertEqual(len(images), 1)
        h, w, c = images[0].shape
        self.assertEqual(h, 200)
        self.assertEqual(w, 300)
        self.assertEqual(c, 3)


if __name__ == '__main__':
    unittest.main()
