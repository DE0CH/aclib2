"""
Created on May 27, 2015

@author: Andre Biedenkapp
"""

import unittest
import numpy
import random
import os
import logging
import epm.experiment_utils.data_handling


class Test(unittest.TestCase):

    def setUp(self):
        self.src_dir = os.path.dirname(os.path.dirname(
                                       os.path.abspath(__file__)))
        self.file_dir = os.path.join(self.src_dir, "files")

        logging.basicConfig(level=logging.DEBUG)
        self.pcs_file = os.path.join(self.file_dir, "pcs",
                                     "parameter_configuration_space_example.pcs")
        self.feature_file = os.path.join(self.file_dir, 'csv',
                                         'example_features.csv')
        self.performance_file = os.path.join(self.file_dir, "csv",
                                             "performance_example.csv")
        self.config_file = os.path.join(self.file_dir, "csv",
                                        "configuration_example.csv")

    @unittest.skip("Don't do matlab data tests")
    def test_data_handling(self):
        epm.experiment_utils.data_handling.loadMatlabData(self.pcs_file,
                                                          self.feature_file,
                                                          self.performance_file,
                                                          self.config_file)

    def test_data_handling_json_downsample(self):
        pcs = os.path.join(self.file_dir, "pcs", "clasp-sat-params-nat.pcs")
        feature = None
        perf = os.path.join(self.file_dir, "json", "n_queens.json")
        encoded_matrix, inst_feat_dict, inst_list, n_feats_used, \
               perf_list, timeout_list, cen_list = \
            epm.experiment_utils.\
                data_handling.load_data(csv_file=perf,
                                        feature_file=None,
                                        pcs_file=pcs,
                                        cutoff=300,
                                        downsample=2,
                                        instance_file=None,
                                        normalize_features=True,
                                        oneHot=True,
                                        par=1, impute_with='def', quality=False,
                                        normalize_parameters=False)
        self.assertEqual(encoded_matrix.shape[0], 2)
        self.assertLessEqual(numpy.max(perf_list), 300)

        self.assertEqual(inst_feat_dict.values()[0], [0])
        self.assertEqual(inst_feat_dict.values()[1], [1])

        self.assertAlmostEqual(perf_list[0], 6.2256899)
        self.assertAlmostEqual(perf_list[1], 0.105933)

    def test_data_handling_json(self):
        pcs = os.path.join(self.file_dir, "pcs", "clasp-sat-params-nat.pcs")
        feature = None
        perf = os.path.join(self.file_dir, "json", "n_queens.json")
        encoded_matrix, inst_feat_dict, inst_list, n_feats_used, \
               perf_list, timeout_list, cen_list = \
            epm.experiment_utils.\
                data_handling.load_data(csv_file=perf,
                                        feature_file=None,
                                        pcs_file=pcs,
                                        cutoff=300,
                                        downsample=None,
                                        instance_file=None,
                                        normalize_features=True,
                                        oneHot=True,
                                        par=1, impute_with='def', quality=False,
                                        normalize_parameters=False)
        self.assertEqual(encoded_matrix.shape[0], 50)
        self.assertLessEqual(numpy.max(perf_list), 300)
        self.assertEqual(inst_feat_dict.values()[0], [13])
        self.assertEqual(inst_feat_dict.values()[1], [1])

        self.assertAlmostEqual(perf_list[0], 300)
        self.assertTrue(timeout_list[0])
        self.assertAlmostEqual(perf_list[2], 261.328, 3)

    def test_data_handling_json_quality(self):
        pcs = os.path.join(self.file_dir, "pcs", "svm_mnist.pcs")
        feature = None
        perf = os.path.join(self.file_dir, "json", "svm_mnist.json")
        encoded_matrix, inst_feat_dict, inst_list, n_feats_used, \
               perf_list, timeout_list, cen_list = \
            epm.experiment_utils.\
                data_handling.load_data(csv_file=perf,
                                        feature_file=None,
                                        pcs_file=pcs,
                                        cutoff=5,
                                        downsample=None,
                                        instance_file=None,
                                        normalize_features=True,
                                        oneHot=True,
                                        par=1, impute_with='def', quality=True,
                                        normalize_parameters=True)
        self.assertEqual(encoded_matrix.shape[0], 50)
        self.assertAlmostEqual(numpy.max(perf_list), 1, 4)
        print(inst_feat_dict)
        self.assertEqual(inst_feat_dict.values()[0], [0])
        self.assertEqual(inst_feat_dict.values()[1], [1])

        self.assertAlmostEqual(perf_list[0], 1, 4)
        self.assertEqual(timeout_list, None)
        self.assertAlmostEqual(perf_list[2], 1, 3)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
