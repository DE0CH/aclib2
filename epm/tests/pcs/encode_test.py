import numpy
import unittest

from epm.pcs.config_space import ConfigSpace


class TestEncode(unittest.TestCase):

    def test_sanity(self):
        X = [[0.5, 1, 0.5], [0.5, 0, 0.5]]
        cat_list = [0, 2, 0]
        self.assertRaises(AssertionError, ConfigSpace.encode, X, cat_list)
        self.assertRaises(AssertionError, ConfigSpace.encode, X, list())

        X = numpy.array(X)
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertListEqual(list(X.shape), [2, 3])
        self.assertListEqual(list(new_x.flatten()),
                             [0.5, 0, 1, 0.5, 0.5, 1, 0, 0.5])

        cat_list = [0, 50, 0]
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertEqual(new_x.shape[1],
                         sum(cat_list) + sum([1 for i in cat_list if i == 0]))

        cat_list = [0, 50, 10]
        self.assertRaises(AssertionError, ConfigSpace.encode, X=X,
                          cat_list=cat_list)

    def test_location(self):
        # middle
        X = numpy.array([[-1, 1, -1], [-2, 0, -2]])
        cat_list = [0, 2, 0]
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertListEqual(list(new_x.flatten()),
                             [-1, 0, 1, -1, -2, 1, 0, -2])

        # left
        X = numpy.array([[1, -1, -1], [0, -2, -2]])
        cat_list = [2, 0, 0]
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertListEqual(list(new_x.flatten()),
                             [0, 1, -1, -1, 1, 0, -2, -2])

        # right
        X = numpy.array([[-1, -1, 1], [-2, -2, 0]])
        cat_list = [0, 0, 2]
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertListEqual(list(new_x.flatten()),
                             [-1, -1, 0, 1, -2, -2, 1, 0])

        # both ende
        X = numpy.array([[0, -1, 1], [0, -2, 0]])
        cat_list = [2, 0, 2]
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertListEqual(list(new_x.flatten()),
                             [1, 0, -1, 0, 1, 1, 0, -2, 1, 0])

        # all
        X = numpy.array([[0, 1, 1], [0, 1, 0]])
        cat_list = [2, 2, 2]
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertListEqual(list(new_x.flatten()),
                             [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

        # zero
        X = numpy.array([[0, 1, 1], [0, 1, 0]])
        cat_list = [0, 0, 0]
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertListEqual(list(new_x.flatten()),
                             [0, 1, 1, 0, 1, 0])

        # middle, next to each other
        X = numpy.array([[-2, 1, 1, -2], [-3, 0, 1, -3]])
        cat_list = [0, 2, 2, 0]
        new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
        self.assertListEqual(list(new_x.flatten()),
                             [-2, 0, 1, 0, 1, -2, -3, 1, 0, 0, 1, -3])

    def test_random(self):
        for i in range(100):
            sample_size = numpy.random.randint(100)+1
            feature_size = numpy.random.randint(100)+1
            X = (numpy.random.randn(sample_size, feature_size)>0.5).astype(int)
            cat_list = list((numpy.random.randn(feature_size)>0.5).astype(int)*2)
            new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
            self.assertEqual(new_x.shape[1], sum(cat_list) + sum([1 for i in cat_list if i == 0]))

if __name__ == '__main__':
    unittest.main()