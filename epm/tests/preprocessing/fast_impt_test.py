"""
Created on Mar 25, 2015

@author: manju
"""
import cProfile
import functools
import itertools
import time
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
from epm.preprocessing.fast_imputor_missing_y import FastImputorY
#from epm.preprocessing.imputor_missing_y import ImputorY


class FastImputorTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.src_dir = os.path.dirname(os.path.dirname(__file__))
        self.nqueens_features = os.path.join(self.src_dir, "files", "csv",
                                             "nqueens-features.csv")
        self.rs = numpy.random.RandomState(1)
        self.pcs_file = os.path.join(self.src_dir, "files", "pcs",
                                     "clasp-sat-params-nat.pcs")
        self.cs = ConfigSpace(self.pcs_file, normalize=False)

        self.data_file = os.path.join(self.src_dir, "files", "csv",
                                 "clasp_queens_cssc14_target_algo_runs.csv")

        self.maxDiff = None

    def test_imputation_nolog(self):
        for par in (1, 10):
            print("##### PAR%d" % par)
            cutoff = 300
            max_data_points = 200

            fp = FeatureParser()
            inst_feat_dict, n_feats_used = \
                fp.read_features(feature_file=self.nqueens_features,
                                 n_feats=1151)

            fpre = PreprocessingFeatures(inst_feats_dict=inst_feat_dict)
            inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)

            pp = PerformanceParser(cutoff=cutoff, par=par, quality=False)
            config_list, perf_list, instance_list, success_list, cen_list = \
                pp.read_data(self.data_file)

            config_matrix = numpy.zeros((len(config_list),
                                         self.cs._n_params+n_feats_used),
                                        dtype=numpy.float32)
            for indx, config in enumerate(config_list):
                config_vec = self.cs.convert_param_dict(config)
                imputed_vec = self.cs.impute_non_active(config_vec, value="def")
                config_matrix[indx] = numpy.hstack((imputed_vec,
                                                    inst_feat_dict[instance_list[indx]]))
            imp = FastImputorY(debug=False)
            #simp = ImputorY(debug=False)

            base = functools.partial(RandomForestRegressor, n_estimators=10,
                                     max_features="sqrt", random_state=1)
            model = functools.partial(bootstrapModel, rng=1, debug=False,
                                      n_bootstrap_samples=20,
                                      bootstrap_size=0.7,
                                      base_estimator=base)

            indices = numpy.array([True]*max_data_points +
                                  [False]*(len(perf_list)-max_data_points),
                                  dtype=bool)
            cen_list = numpy.array(cen_list)
            instance_list = numpy.array(instance_list)
            perf_list = numpy.array(perf_list)
            new_dur = []
            #old_dur = []
            for i in range(10):
                self.assertEqual(sum(perf_list[indices] > cutoff*par), 0)
                print("iteration %d" % i)
                self.rs.shuffle(indices)
                start = time.time()
                new_ys = imp.impute_y(y=perf_list[indices],
                                      is_censored=cen_list[indices],
                                      cutoff=cutoff, par=par,
                                      configs_list=numpy.array(config_matrix)[indices],
                                      inst_list=instance_list[indices],
                                      inst_feature_dict=inst_feat_dict,
                                      model=model, log=False)
                new_dur.append(time.time()-start)
                self.assertEqual(sum(numpy.array(new_ys) > cutoff*par), 0)

                """
                start = time.time()
                snew_ys = simp.impute_y(y=perf_list[indices],
                                        is_censored=cen_list[indices],
                                        cutoff=cutoff, par=par,
                                        configs_list=numpy.array(config_matrix)[indices],
                                        inst_list=instance_list[indices],
                                        inst_feature_dict=inst_feat_dict,
                                        model=model, log=False)
                old_dur.append(time.time()-start)
                self.assertEqual(sum(numpy.array(snew_ys) > cutoff*par), 0)
                self.assertListEqual(new_ys, snew_ys)
                """
                for idx, y in enumerate(new_ys):
                    if not (cen_list[indices])[idx]:
                        # Point is the same
                        self.assertAlmostEqual(y, (perf_list[indices])[idx])
                        #self.assertAlmostEqual(snew_ys[idx], (perf_list[indices])[idx])
                    elif (perf_list[indices])[idx] > cutoff:
                        self.assertEqual(y, (perf_list[indices])[idx])
                        #self.assertEqual(snew_ys[idx], (perf_list[indices])[idx])
                    else:
                        self.assertLessEqual(y, cutoff*par)
                        self.assertGreater(y, (perf_list[indices])[idx])
                        #self.assertLessEqual(snew_ys[idx], cutoff*par)
                        #self.assertGreater(snew_ys[idx], (perf_list[indices])[idx])

    def test_imputation_log(self):
        for par in (10,):
            print("##### PAR%d" % par)
            cutoff = 300
            max_data_points = 200

            fp = FeatureParser()
            inst_feat_dict, n_feats_used = \
                fp.read_features(feature_file=self.nqueens_features,
                                 n_feats=1151)

            fpre = PreprocessingFeatures(inst_feats_dict=inst_feat_dict)
            inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)

            pp = PerformanceParser(cutoff=cutoff, par=par)
            config_list, perf_list, instance_list, success_list, cen_list = \
                pp.read_data(self.data_file)

            config_matrix = numpy.zeros((len(config_list),
                                         self.cs._n_params+n_feats_used),
                                        dtype=numpy.float32)
            for indx, config in enumerate(config_list):
                config_vec = self.cs.convert_param_dict(config)
                imputed_vec = self.cs.impute_non_active(config_vec, value="def")
                config_matrix[indx] = numpy.hstack((imputed_vec,
                                                    inst_feat_dict[instance_list[indx]]))
            imp = FastImputorY(debug=False)
            #simp = ImputorY(debug=False)

            base = functools.partial(RandomForestRegressor, n_estimators=10,
                                     max_features="sqrt", random_state=1)
            model = functools.partial(bootstrapModel, rng=1, debug=False,
                                      n_bootstrap_samples=20, bootstrap_size=0.7,
                                      base_estimator=base)

            indices = numpy.array([True]*max_data_points +
                                  [False]*(len(perf_list)-max_data_points),
                                  dtype=bool)
            cen_list = numpy.array(cen_list)
            instance_list = numpy.array(instance_list)
            perf_list = numpy.array(perf_list)
            new_dur = []
            #old_dur = []
            for i in range(10):
                self.assertEqual(sum(perf_list[indices] > cutoff*par), 0)
                print("iteration %d" % i)
                self.rs.shuffle(indices)
                start = time.time()
                new_ys = imp.impute_y(y=perf_list[indices],
                                      is_censored=cen_list[indices],
                                      cutoff=cutoff, par=par,
                                      configs_list=numpy.array(config_matrix)[indices],
                                      inst_list=instance_list[indices],
                                      inst_feature_dict=inst_feat_dict,
                                      model=model, log=True)
                new_dur.append(time.time()-start)
                self.assertEqual(sum(numpy.array(new_ys) > cutoff * par), 0)
                """
                start = time.time()
                snew_ys = simp.impute_y(y=perf_list[indices],
                                        is_censored=cen_list[indices],
                                        cutoff=cutoff, par=par,
                                        configs_list=numpy.array(config_matrix)[indices],
                                        inst_list=instance_list[indices],
                                        inst_feature_dict=inst_feat_dict,
                                        model=model, log=True)
                old_dur.append(time.time()-start)
                self.assertEqual(sum(numpy.array(snew_ys) > 10**numpy.log10(cutoff*par)+1e-3), 0)
                [self.assertAlmostEqual(new_ys[i], snew_ys[i], 3) if new_ys[i] < cutoff else 0 for i in range(len(snew_ys))]
                """
                for idx, y in enumerate(new_ys):
                    if not (cen_list[indices])[idx]:
                        # Point is the same
                        self.assertAlmostEqual(y, (perf_list[indices])[idx])
                        #self.assertAlmostEqual(snew_ys[idx], (perf_list[indices])[idx])
                    elif (perf_list[indices])[idx] > cutoff:
                        self.assertAlmostEqual(y, numpy.power(10, (numpy.log10(cutoff))))  # ~ 299.999999
                        #self.assertAlmostEqual(snew_ys[idx], numpy.power(10, (numpy.log10(cutoff))))  # ~ 299.999999
                    else:
                        if y > cutoff:
                            self.assertGreater(y, cutoff, 2)
                            #self.assertGreater(snew_ys[idx], cutoff, 2)
                        else:
                            self.assertLess(y, cutoff)
                        self.assertGreater(y, (perf_list[indices])[idx])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
