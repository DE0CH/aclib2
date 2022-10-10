"""
Created on Mar 25, 2015

@author: lindauer
"""

import unittest
import random
import os
import logging

from epm.reader.performance_parser import PerformanceParser
from epm.pcs.config_space import ConfigSpace
from epm.reader.feature_parser import FeatureParser


class Test(unittest.TestCase):

    def setUp(self):
        random.seed(12345)
        self.src_dir = os.path.dirname(os.path.dirname(__file__))
        logging.basicConfig(level=logging.DEBUG)

    def test_clasp_cssc14_queens_perf(self):
        
        data_file = os.path.join(self.src_dir, "files", "csv",
                                 "clasp_queens_cssc14_target_algo_runs.csv")

        pp = PerformanceParser(cutoff=300, par=1, quality=False)
        config_list, perf_list, instance_list, success_list, cen_list = \
            pp.read_data(data_file)

        # test for reading multiple files
        cl_1, pl_1, il_1, sl_1, cenl_1 = pp.\
            read_data_from_multiple_files(files_=[data_file, ])

        self.assertListEqual(config_list, cl_1)
        self.assertListEqual(perf_list, pl_1)
        self.assertListEqual(instance_list, il_1)
        self.assertListEqual(success_list, sl_1)
        self.assertListEqual(cen_list, cenl_1)
        
        pcs_file = os.path.join(self.src_dir, "files", "pcs",
                                "clasp-sat-params-nat.pcs")

        cs = ConfigSpace(pcs_file, normalize=False)
        
        for conf in config_list:
            cs.convert_param_dict(conf)
            
    def test_cssc14_queens_feats(self):
        feat_file = os.path.join(self.src_dir, "files", "csv",
                                 "nqueens-features.csv")
        
        fp = FeatureParser()
        # only 119 features available
        fp.read_features(feature_file=feat_file, n_feats=1151)

if __name__ == "__main__":
    unittest.main()