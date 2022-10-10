"""
Created on Mar 27, 2015

@author: Katharina Eggensperger
"""
import logging
import unittest
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import epm.preprocessing.pre_feature
import epm.models.external_rfr_predict
import epm.reader.feature_parser
import epm.reader.performance_parser
import epm.pcs.config_space


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("bootstrapModelTest")


class RunForestRunPredictTest(unittest.TestCase):

    def setUp(self):
        self.rng = 1
        self.rs = numpy.random.RandomState(self.rng)
        self.src_dir = os.path.dirname(os.path.dirname(__file__))
        self.nqueens_features = os.path.join(self.src_dir, "files", "csv",
                                             "nqueens-features.csv")
        self.model_fl = os.path.join(self.src_dir, "rfr.mdl.pkl")
        self.cs_file = os.path.join(self.src_dir, "files", "pcs",
                                "clasp-sat-params-nat.pcs")
        self.cs = epm.pcs.config_space.ConfigSpace(self.cs_file, normalize=True)

    def test_invalidInput(self):
        m = epm.models.external_rfr_predict.ExternalRFRPredict(cs=self.cs,
                                                               n_feats=119,
                                                               seed=self.rng,
                                                               debug=False,
                                                               cutoff=300,
                                                               threshold=300*10)
        y = self.rs.randn(100, 1).flatten()
        X = numpy.vstack([y**2, y+7, self.rs.randn(100, 1).flatten()]).T
        m.types = numpy.array([0]*X.shape[1], dtype=numpy.uint64)
        y = y.astype(numpy.float32)

        X = X.astype(numpy.float32)
        m.fit(X=X, y=y)
        pred = m.predict(X=X[:10, :], logged=False)
        self.assertEqual(pred.ndim, 1)
        self.assertEqual(len(pred.shape), 1)
        self.assertEqual(pred.shape[0], 10)

        X = X.astype(numpy.float64)
        self.assertRaises(ValueError, m.fit, X=X, y=y)

    def test_real_data_nonpar(self):
        fp = epm.reader.feature_parser.FeatureParser()
        inst_feat_dict, n_feats_used = fp.read_features(
                feature_file=self.nqueens_features, n_feats=1151)

        fpre = epm.preprocessing.pre_feature.PreprocessingFeatures(
                inst_feats_dict=inst_feat_dict)
        inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)

        data_file = os.path.join(self.src_dir, "files", "csv",
                                 "clasp_queens_cssc14_target_algo_runs.csv")

        pp = epm.reader.performance_parser.PerformanceParser(cutoff=300, par=1)
        config_list, perf_list, instance_list, success_list, cen_list = pp.\
            read_data(data_file)

        config_matrix = numpy.zeros((len(config_list),
                                     self.cs._n_params+n_feats_used),
                                    dtype=numpy.float32)
        for indx, config in enumerate(config_list):
            config_vec = self.cs.convert_param_dict(config)
            # mean imputation
            imputed_vec = self.cs.impute_non_active(config_vec, value="def")
            config_matrix[indx] = numpy.hstack((imputed_vec,
                                                inst_feat_dict[instance_list[indx]]))
        perf_list = numpy.log10(numpy.array(perf_list).flatten())
        cutoff = numpy.log10(300)
        threshold = numpy.log10(300)

        kf = KFold(len(perf_list), fold=20)
        rmse = list()
        for train, test in kf:
            x_train = config_matrix[train]
            y_train = perf_list[train]
            m = epm.models.external_rfr_predict.\
                ExternalRFRPredict(seed=self.rng, debug=False, cs=self.cs,
                                   n_feats=n_feats_used, cutoff=cutoff,
                                   threshold=threshold)
            m.fit(x_train, y_train)

            x_test = config_matrix[test]
            y_test = perf_list[test]
            pred = m.predict(x_test, logged=True)

            score = numpy.sqrt(mean_squared_error(y_pred=pred, y_true=y_test))
            rmse.append(score)
        print(rmse, numpy.mean(rmse))
        self.assertLess(numpy.mean(rmse), 0.6)

    def test_real_data_par(self):
        fp = epm.reader.feature_parser.FeatureParser()
        inst_feat_dict, n_feats_used = fp.read_features(
                feature_file=self.nqueens_features, n_feats=1151)

        fpre = epm.preprocessing.pre_feature.PreprocessingFeatures(
                inst_feats_dict=inst_feat_dict)
        inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)

        data_file = os.path.join(self.src_dir, "files", "csv",
                                 "clasp_queens_cssc14_target_algo_runs.csv")

        pp = epm.reader.performance_parser.PerformanceParser(cutoff=300, par=10)
        config_list, perf_list, instance_list, success_list, cen_list = pp.\
            read_data(data_file)

        config_matrix = numpy.zeros((len(config_list),
                                     self.cs._n_params+n_feats_used),
                                    dtype=numpy.float32)
        for indx, config in enumerate(config_list):
            config_vec = self.cs.convert_param_dict(config)
            # mean imputation
            imputed_vec = self.cs.impute_non_active(config_vec, value="def")
            config_matrix[indx] = numpy.hstack((imputed_vec,
                                                inst_feat_dict[instance_list[indx]]))
        perf_list = numpy.log10(numpy.array(perf_list).flatten())
        cutoff = numpy.log10(300)
        threshold = numpy.log10(3000)

        kf = KFold(len(perf_list), n_folds=20)
        rmse = list()
        for train, test in kf:
            x_train = config_matrix[train]
            y_train = perf_list[train]
            m = epm.models.external_rfr_predict.\
                ExternalRFRPredict(seed=self.rng, debug=False, cs=self.cs,
                                   n_feats=n_feats_used, cutoff=cutoff,
                                   threshold=threshold)
            m.fit(x_train, y_train)

            x_test = config_matrix[test]
            y_test = perf_list[test]
            pred = m.predict(x_test, logged=True)
            pred[pred >= cutoff] = threshold

            score = numpy.sqrt(mean_squared_error(y_pred=pred, y_true=y_test))
            rmse.append(score)
        print(rmse, numpy.mean(rmse))
        self.assertLess(numpy.mean(rmse), 0.63)

    def pickle_test(self):
        fp = epm.reader.feature_parser.FeatureParser()
        inst_feat_dict, n_feats_used = fp.read_features(
                feature_file=self.nqueens_features, n_feats=1151)

        fpre = epm.preprocessing.pre_feature.PreprocessingFeatures(
                inst_feats_dict=inst_feat_dict)
        inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)

        data_file = os.path.join(self.src_dir, "files", "csv",
                                 "clasp_queens_cssc14_target_algo_runs.csv")

        pp = epm.reader.performance_parser.PerformanceParser(cutoff=300, par=10)
        config_list, perf_list, instance_list, success_list, cen_list = \
            pp.read_data(data_file)

        config_matrix = numpy.zeros((len(config_list),
                                     self.cs._n_params+n_feats_used),
                                    dtype=numpy.float32)
        for indx, config in enumerate(config_list):
            config_vec = self.cs.convert_param_dict(config)
            imputed_vec = self.cs.impute_non_active(config_vec, value="def")
            config_matrix[indx] = numpy.\
                hstack((imputed_vec, inst_feat_dict[instance_list[indx]]))
        perf_list = numpy.log10(numpy.array(perf_list).flatten())

        m = epm.models.external_rfr_predict.\
            ExternalRFRPredict(seed=self.rng, debug=False, cs=self.cs,
                               n_feats=n_feats_used, cutoff=numpy.log10(300),
                               threshold=numpy.log10(3000))
        m.fit(X=config_matrix[1:, :], y=perf_list[1:])
        test_data = config_matrix[0].flatten()
        a_pred = m.predict(test_data,
                           prediction_threshold=0,
                           cutoff=numpy.log10(300),
                           threshold=numpy.log10(300*10))

        # Now pickle model
        pickle.dump(obj=m,
                    file=open(self.model_fl, 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)

        del m

        m1 = pickle.load(open(self.model_fl, 'wb'))
        b_pred = m1.predict(test_data, prediction_threshold=0,
                            cutoff=numpy.log10(300),
                            threshold=numpy.log10(300*10))
        self.assertAlmostEqual(a_pred, b_pred)

    def tearDown(self):
        if os.path.exists(self.model_fl):
            os.remove(self.model_fl)

if __name__ == "__main__":
    unittest.main()
