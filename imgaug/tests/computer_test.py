import unittest
from imgaug import computer, image


class TestComputer(unittest.TestCase):
    def test_persister(self):
        pers = computer.Persister('resources/bag.png', augmenter=image.Augmenter(0.1))
        self.assertEqual(pers.label, 'bag')
        self.assertEqual(pers.ext, 'png')

    def test_persister_call(self):
        pers = computer.Persister('resources/bag.png', augmenter=image.Augmenter(0.01), label='gucci_bag')
        files = list(pers)
        label, filepath = files[-1]
        self.assertEqual(label, 'gucci_bag')
        self.assertEqual(filepath, 'gucci_bag_0011.png')


if __name__ == '__main__':
    unittest.main()