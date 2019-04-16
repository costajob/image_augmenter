import unittest
from imgaug import computer, image


class TestComputer(unittest.TestCase):
    def test_persister(self):
        pers = computer.Persister('resources/bag.png', augmenter=image.Augmenter(0.1))
        self.assertEqual(pers.label, 'bag')
        self.assertEqual(pers.ext, 'png')

    def test_persister_stream(self):
        with open('resources/bag.png', 'rb') as f:
            pers = computer.Persister('resources/bag.png', f, augmenter=image.Augmenter(0.01), label='gucci_bag')
            files = list(pers)
            label, filepath = files[-1]
            self.assertEqual(label, 'gucci_bag')
            self.assertEqual(filepath, 'gucci_bag_005.png')

    def test_zipper(self):
        zipper = computer.Zipper('resources', normalizer=image.Normalizer(16), augmenter=image.Augmenter(.05))
        for _, archive in zipper:
            if 'shirt' in archive:
                self.assertTrue(archive.endswith('.jpg'))
            else:
                self.assertTrue(archive.endswith('.png'))

if __name__ == '__main__':
    unittest.main()
