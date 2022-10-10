"""
Created on Mar 25, 2015

@author: manju
"""
import functools
import unittest
import logging
import os

import numpy

from sklearn.ensemble import RandomForestRegressor

from epm.models.bootstrapModel import bootstrapModel
from epm.reader.feature_parser import FeatureParser
from epm.preprocessing.pre_feature import PreprocessingFeatures
from epm.reader.performance_parser import PerformanceParser
from epm.pcs.config_space import ConfigSpace
from epm.preprocessing.fast_imputor_missing_y import FastImputorY as ImputorY


class pre_feature_Test(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.src_dir = os.path.dirname(os.path.dirname(__file__))
        self.nqueens_features = os.path.join(self.src_dir, "files", "csv",
                                             "nqueens-features.csv")
        self.data_file = os.path.join(self.src_dir, "files", "csv",
                                      "clasp_queens_cssc14_target_algo_runs.csv")
        self.pcs_file = os.path.join(self.src_dir, "files", "pcs",
                                     "clasp-sat-params-nat.pcs")

        self.fp = FeatureParser()
        a, b = self.fp.read_features(feature_file=self.nqueens_features, n_feats=1151)
        self.inst_feat_dict = a
        self.n_feats_used = b
        self.cutoff = 300
        self.max_data_points = 300
        self._dtype = numpy.float32

    def test_nqueens_feats(self):

        fpre = PreprocessingFeatures(inst_feats_dict=self.inst_feat_dict)
        inst_feat_dict = fpre.normalization(inst_feats_dict=self.inst_feat_dict)
        logging.info(self.inst_feat_dict[self.inst_feat_dict.keys()[0]])
        
    def test_imputation(self):
        fpre = PreprocessingFeatures(inst_feats_dict=self.inst_feat_dict)
        inst_feat_dict = fpre.normalization(inst_feats_dict=self.inst_feat_dict)

        pp = PerformanceParser(cutoff=self.cutoff, par=1, dtype=numpy.float32)
        config_list, perf_list, instance_list, success_list, cen_list = \
            pp.read_data(self.data_file)
        print(cen_list)
        cs = ConfigSpace(self.pcs_file, normalize=False)
        
        config_matrix = numpy.zeros((len(config_list), cs._n_params),
                                    dtype=numpy.float32)
        for indx, config in enumerate(config_list):
            config_vec = cs.convert_param_dict(config)
            imputed_vec = cs.impute_non_active(config_vec, value="def")
            config_matrix[indx] = imputed_vec

        imp = ImputorY(debug=True)

        base = functools.partial(RandomForestRegressor, n_estimators=10,
                                 random_state=1)
        model = functools.partial(bootstrapModel, rng=1, debug=False,
                                  n_bootstrap_samples=20, bootstrap_size=0.7,
                                  base_estimator=base)


        perf_list = numpy.array(perf_list, dtype=self._dtype)
        config_list = numpy.array(config_matrix, numpy.float32)

        new_ys = imp.impute_y(y=perf_list[:self.max_data_points],
                              is_censored=cen_list[:self.max_data_points],
                              cutoff=self.cutoff,
                              configs_list=config_list[:self.max_data_points],
                              inst_list=instance_list[:self.max_data_points],
                              inst_feature_dict=self.inst_feat_dict,
                              model=model, log=False)

        for idx, y in enumerate(new_ys):
            if not cen_list[idx]:
                # Point is the same
                self.assertAlmostEqual(y, perf_list[idx])
            elif perf_list[idx] > self.cutoff:
                # This was wrong before
                self.assertEqual(y, perf_list[idx])
            else:
                self.assertLessEqual(y, self.cutoff)
                self.assertGreater(y, perf_list[idx])

        sec_new_ys = imp.impute_y(y=perf_list[:self.max_data_points],
                                  is_censored=cen_list[:self.max_data_points],
                                  cutoff=self.cutoff,
                                  configs_list=config_list[:self.max_data_points],
                                  inst_list=instance_list[:self.max_data_points],
                                  inst_feature_dict=self.inst_feat_dict,
                                  model=model, log=True)

        for idx, y in enumerate(sec_new_ys):
            if not cen_list[idx]:
                # Point is the same
                self.assertAlmostEqual(y, perf_list[idx])
            elif perf_list[idx] > self.cutoff:
                self.assertAlmostEqual(y, numpy.power(10, (numpy.log10(self.cutoff))))  # ~ 299.999999
            else:
                if y > self.cutoff:
                    print(y, self.cutoff)
                    self.assertAlmostEqual(y, self.cutoff, 3)
                else:
                    self.assertLess(y, self.cutoff)
                self.assertGreater(y, perf_list[idx])