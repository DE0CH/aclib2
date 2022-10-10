import numpy
import unittest
import random
import os
import time

from epm.pcs.config_space import ConfigSpace


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        random.seed(12345)
        self.src_dir = os.path.dirname(os.path.dirname(__file__))
        self.clasp_pcs_file = os.path.join(self.src_dir, "files", "pcs",
                                           "clasp-sat-params-nat.pcs")
        self.lingeling_pcs_file = os.path.join(self.src_dir, "files", "pcs",
                                               "lingeling-params.pcs")

        #logging.basicConfig(level=logging.DEBUG)

    def test_clasp(self):
        
        print("clasp")

        cs = ConfigSpace(self.clasp_pcs_file, normalize=True)
        
        t0 = time.time()
        for _ in range(1,10000):
            vec = cs.get_random_config_vector()
            #print(cs.convert_param_vector(vec))
        print("clasp random configs time (sec): %f" %(time.time() - t0))
        assert len(cs.parameters) == 75, "#params: %d should be 75" %(len(cs.parameters)) 

    def test_convert_clasp(self):
        
        print("clasp convert")
        cs = ConfigSpace(self.clasp_pcs_file, normalize=True)
        def_config = cs.get_default_config_dict()
        print(def_config)
        def_vec = cs.convert_param_dict(def_config)
        def_config_back = cs.convert_param_vector(def_vec)
        #print(def_config)
        #print(def_vec)
        
        for param, value in def_config.iteritems():
            assert value == def_config_back[param], "%s: %s vs %s" %(param, str(value), str(def_config_back[param]))
        
    def test_neighbor_clasp(self):
        print("clasp neighbor")

        cs = ConfigSpace(self.clasp_pcs_file, normalize=True)
        def_config = cs.get_default_config_dict()
        def_vec = cs.convert_param_dict(def_config)
        
        
        t0 = time.time()
        for _ in range(0,10000):
            neighbor_vec = cs.get_random_neighbor(def_vec)
        print("clasp neighbor time (sec): %f" %(time.time() - t0))
        
        #print(def_vec)
        #print(neighbor_vec)
        #=======================================================================
        # neighbor_dict = cs.convert_param_vector(neighbor_vec)
        # for p in def_config.keys():
        #     if neighbor_dict.get(p, None) != def_config[p]:
        #         print(p, neighbor_dict.get(p, None), def_config[p])
        #=======================================================================
        
    def test_lingeling(self):
        
        print("lingeling")
        
        cs = ConfigSpace(self.lingeling_pcs_file, normalize=True)
        
        t0 = time.time()
        for _ in range(1,10000):
            cs.get_random_config_vector()
        print("lingeling random configs time (sec): %f" %(time.time() - t0))
        
        assert len(cs.parameters) == 323, "#params: %d should be 75" %(len(cs.parameters)) 
        
    def test_convert_lingeling(self):
        
        print("lingeling convert")
        cs = ConfigSpace(self.lingeling_pcs_file, normalize=True)
        def_config = cs.get_default_config_dict()
        def_vec = cs.convert_param_dict(def_config)
        def_config_back = cs.convert_param_vector(def_vec)
        #print(def_config)
        #print(def_vec)
        
        for param, value in def_config.iteritems():
            assert value == def_config_back[param], "%s: %s vs %s" %(param, str(value), str(def_config_back[param]))

    def test_neighbor_lingeling(self):
        print("lingeling neighbor")

        cs = ConfigSpace(self.lingeling_pcs_file, normalize=True)
        def_config = cs.get_default_config_dict()
        def_vec = cs.convert_param_dict(def_config)
        
        
        t0 = time.time()
        for _ in range(0,10000):
            neighbor_vec = cs.get_random_neighbor(def_vec)
        print("lingeling neighbor time (sec): %f" %(time.time() - t0))
        
        #print(def_vec)
        #print(neighbor_vec)
        #=======================================================================
        # neighbor_dict = cs.convert_param_vector(neighbor_vec)
        # for p in def_config.keys():
        #     if neighbor_dict.get(p, None) != def_config[p]:
        #         print(p, neighbor_dict.get(p, None), def_config[p])
        #=======================================================================

    def encode_test_sanity(self):
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
        self.assertEqual(new_x.shape[1], sum(cat_list) +
                         sum([1 for i in cat_list if i == 0]))

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
        self.assertListEqual(list(new_x.flatten()), [0, 1, 1, 0, 1, 0])

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
            X = (numpy.random.randn(sample_size, feature_size) > 0.5).\
                astype(int)
            cat_list = list((numpy.random.randn(feature_size) > 0.5).
                            astype(int)*2)
            new_x = ConfigSpace.encode(X=X, cat_list=cat_list)
            self.assertEqual(new_x.shape[1], sum(cat_list) +
                             sum([1 for i in cat_list if i == 0]))

if __name__ == '__main__':
    unittest.main()