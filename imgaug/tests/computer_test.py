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

    def test_zipper(self):
        zipper = computer.Zipper('resources', size=8, cutoff=.01)
        for accumulator in zipper:
            for _, archive in accumulator:
                if 'shirt' in archive:
                    self.assertTrue(archive.endswith('.jpg'))
                else:
                    self.assertTrue(archive.endswith('.png'))

    def test_zipper_x_zip(self):
        zipper = computer.Zipper('resources', x_zip=15, size=8, cutoff=.01)
        for i, accumulator in enumerate(zipper):
            assert len(accumulator) <= 15
        self.assertEqual(i, 3)


if __name__ == '__main__':
    unittest.main()
