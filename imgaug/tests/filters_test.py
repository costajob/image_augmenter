import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from imgaug import filters


class TestFilters(unittest.TestCase):
    def setUp(self):
        img = filters.Image.open('resources/shirt.jpg')
        self.img = filters.np.asarray(img)

    def test_blur(self):
        with patch.object(filters, 'uniform_filter') as mocked:
            f = filters.Blur()
            f(self.img, .1)
            mocked.assert_called_with(self.img, size=(.1, .1, 1))

    def test_flip(self):
        f = filters.Flip()
        self.assertTrue(callable(f))

    def test_gamma(self):
        with patch.object(filters, 'adjust_gamma') as mocked:
            f = filters.Gamma()
            f(self.img, .1)
            mocked.assert_called_with(self.img, gamma=.1, gain=.9)

    def test_gaussian(self):
        with patch.object(filters, 'gaussian_filter') as mocked:
            f = filters.Gaussian()
            f(self.img, .1)
            mocked.assert_called_with(self.img, .1)

    def test_noise(self):
        with patch.object(filters, 'random_noise') as mocked:
            f = filters.Noise()
            f(self.img, .1)
            mocked.assert_called_with(self.img, mode='speckle', var=.1)

    def test_rescale(self):
        with patch.object(filters, 'rescale', return_value=self.img*1.5) as mocked:
            f = filters.Rescale()
            f(self.img, 1.5)
            mocked.assert_called_with(self.img, 1.5, mode='constant', anti_aliasing=True, multichannel=True)

    def test_rotate(self):
        with patch.object(filters, 'rotate') as mocked:
            f = filters.Rotate()
            f(self.img, -60)
            mocked.assert_called_with(self.img, -60, cval=1, mode='edge')

    def test_shift_valid(self):
        h, w, _ = [d // 2 for d in self.img.shape]
        self.assertFalse(filters.Shift()._valid(self.img, min(w, h)))
        self.assertFalse(filters.Shift('h')._valid(self.img, w))
        self.assertFalse(filters.Shift('v')._valid(self.img, h))

    def test_shift_vectors(self):
        self.assertEqual(filters.Shift()._vector(10), (10, 10))
        self.assertEqual(filters.Shift('h')._vector(10), (10, 0))
        self.assertEqual(filters.Shift('v')._vector(10), (0, 10))

    def test_shift(self):
        with patch.object(filters, 'warp') as mocked, patch.object(filters, 'AffineTransform', return_value=(10, 10)) as tr:
            f = filters.Shift()
            f(self.img, 10)
            mocked.assert_called_with(self.img, inverse_map=tr(), mode='edge')

    def test_skew(self):
        with patch.object(filters, 'warp') as mocked, patch.object(filters, 'AffineTransform', return_value=.3) as tr:
            f = filters.Skew()
            f(self.img, .3)
            mocked.assert_called_with(self.img, inverse_map=tr(), mode='edge')

    def test_pixel(self):
        f = filters.Pixel('max')
        self.assertEqual(f.filter.__class__, filters.MaxFilter.__class__)
        self.assertTrue(callable(f))

    def test_pixel_oversizes(self):
        f = filters.Pixel('max')
        img = filters.np.resize(self.img, (64, 64, 3))
        self.assertIsNone(f(img, 5))
        self.assertIsNotNone(f(img, 3))

    def test_unsharp(self):
        f = filters.Unsharp()
        self.assertTrue(callable(f))


if __name__ == '__main__':
    unittest.main()
