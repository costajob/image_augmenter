import unittest
from imgaug import image


class TestImage(unittest.TestCase):
    def test_labeller_gucci(self):
        lbl = image.Labeller()
        label = lbl('resources/431665_LUFAD_8888_002_073_0000_Light.png')
        self.assertEqual(label, '431665LUFAD8888')

    def test_labeller_mmfg(self):
        lbl = image.Labeller()
        label = lbl('resources/1096023906001-c-suit-veletta-albino.jpg')
        self.assertEqual(label, '1096023906001')

    def test_labeller_tods(self):
        lbl = image.Labeller()
        label = lbl('resources/XXM56A0V430JK4V814-01.jpg')
        self.assertEqual(label, 'XXM56A0V430JK4V814')

    def test_labeller_custom(self):
        lbl = image.Labeller(digits=8)
        label = lbl('resources/product_a_1.JPG')
        self.assertEqual(label, 'producta')

    def test_labeller_plain(self):
        lbl = image.Labeller()
        label = lbl('resources/80038726.jpg')
        self.assertEqual(label, '80038726')

    def test_normalization_path(self):
        norm = image.Normalizer(size=64)
        img = norm('resources/bag.png')
        self.assertEqual(img.shape, (42, 64, 4))

    def test_normalization_stream(self):
        norm = image.Normalizer(size=64)
        with open('resources/bag.png', 'rb') as f:
            img = norm(f)
            self.assertEqual(img.shape, (42, 64, 4))

    def test_normalization_canvas(self):
        norm = image.Normalizer(size=64, canvas=True)
        img = norm('resources/bag.png')
        self.assertEqual(img.shape, (64, 64, 4))

    def test_normalization_colored_canvas(self):
        norm = image.Normalizer(size=64, canvas='FF0000')
        img = norm('resources/bag.png')
        self.assertEqual(img.shape, (64, 64, 4))

    def test_normalization_bkg_canvas(self):
        norm = image.Normalizer(size=64, canvas='resources/office.png')
        img = norm('resources/bag.png')
        self.assertEqual(img.shape, (64, 64, 4))
   
    def test_augmenting_attributes(self):
        aug = image.Augmenter()
        self.assertEqual(len(aug.transformers), 7)
        self.assertEqual(aug.count, 1000)
    
    def test_augmenting(self):
        aug = image.Augmenter(.01)
        images = list(aug('resources/bag.png'))
        self.assertEqual(len(images), aug.count)
        for img in images:
            self.assertEqual(img.shape, (200, 300, 3))

    def test_augmenting_skip(self):
        aug = image.Augmenter(0)
        images = list(aug('resources/bag.png'))
        self.assertEqual(aug.count, 1)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (200, 300, 3))


if __name__ == '__main__':
    unittest.main()
