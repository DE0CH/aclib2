"""
Created on Mar 27, 2015

@author: Katharina Eggensperger
"""
import functools
import logging
import unittest
import os

import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import epm.preprocessing.fast_imputor_missing_y
import epm.preprocessing.pre_feature
import epm.models.bootstrapModel as epmbootstrap
import epm.reader.feature_parser
import epm.reader.performance_parser
import epm.pcs.config_space


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("bootstrapModelTest")


class bootstrapModelTest(unittest.TestCase):

    def setUp(self):
        self.rng = 12345
        self.rs = numpy.random.RandomState(self.rng)
        self.src_dir = os.path.dirname(os.path.dirname(__file__))
        self.nqueens_features = os.path.join(self.src_dir, "files", "csv",
                                             "nqueens-features.csv")

    def test_invalidInput(self):
        base = functools.partial(RandomForestRegressor, n_estimators=10,
                                 random_state=self.rng)
        m = epmbootstrap.bootstrapModel(rng=self.rng, debug=True,
                                        n_bootstrap_samples=50,
                                        bootstrap_size=0.7,
                                        base_estimator=base)
        y = self.rs.randn(100, 1).flatten()
        X = numpy.vstack([y**2, y+7, self.rs.randn(100, 1).flatten()]).T

        X[1, 1] = numpy.nan
        self.assertRaises(ValueError, m.fit, X=X, y=y)

        X[1, 1] = numpy.inf
        self.assertRaises(ValueError, m.fit, X=X, y=y)

        X[1, 1] = None
        self.assertRaises(ValueError, m.fit, X=X, y=y)

    def test_RandomForest(self):
        base = functools.partial(RandomForestRegressor, n_estimators=10,
                                 random_state=self.rng)
        m = epmbootstrap.bootstrapModel(rng=self.rng, debug=True,
                                        n_bootstrap_samples=50,
                                        bootstrap_size=0.7,
                                        base_estimator=base)

        y = self.rs.randn(100, 1).flatten()
        X = numpy.vstack([y**2, y+7, self.rs.randn(100).flatten()]).T
        y = y.astype(numpy.float32)
        X = X.astype(numpy.float32)

        m.fit(X=X, y=y)
        test_data = numpy.array([4.19052259e-02, 6.79529234e+00,
                                 1.00718936e+00],
                                dtype=numpy.float32).reshape([1, 3])
        p_mean, p_std = m.predict_mean_std(test_data)
        # NOTE: Updated March 2018 from -0.12918793, 0.08290447
        self.assertAlmostEqual(-0.13399015, p_mean)  # -0.20470766
        self.assertAlmostEqual(0.08856845, p_std)

    def test_LinearRegression(self):
        base = LinearRegression
        m = epmbootstrap.bootstrapModel(rng=self.rng, debug=True,
                                        n_bootstrap_samples=50,
                                        bootstrap_size=0.7,
                                        base_estimator=base)

        y = self.rs.randn(100).flatten()
        X = numpy.vstack([y**2, y+7, self.rs.randn(100, 1).flatten()]).T
        y = y.astype(numpy.float32)
        X = X.astype(numpy.float32)
        m.fit(X=X, y=y)

        test_data = numpy.array([4.19052259e-02, 6.79529234e+00,
                                 1.00718936e+00],
                                dtype=numpy.float32).reshape([1, 3])
        p_mean, p_std = m.predict_mean_std(test_data)
        self.assertAlmostEqual(-0.20470791, p_mean)  # -0.20470766
        self.assertAlmostEqual(6.39342204e-07, p_std)

    def test_random(self):
        base = RandomForestRegressor  # doesn't matter
        for i in range(50):
            n_data = numpy.random.randint(low=100, high=300)
            n_feat = numpy.random.randint(low=10, high=50)
            n_bootstrap_samples = self.rs.randint(low=5, high=20)
            bootstrap_size = self.rs.randint(low=10, high=99)/100.0

            y = self.rs.randn(n_data, 1).flatten()
            X = numpy.vstack([y**2, y+7, self.rs.randn(n_feat-2, n_data)]).T
            y = y.astype(numpy.float32)
            X = X.astype(numpy.float32)

            m = epmbootstrap.\
                bootstrapModel(rng=self.rng, debug=True,
                               n_bootstrap_samples=n_bootstrap_samples,
                               bootstrap_size=bootstrap_size,
                               base_estimator=base)
            m.fit(X=X, y=y)
            m, s = m.predict_mean_std(X[:10, :])
            [self.assertTrue(si > 0) for si in s]
            del X, y

    def test_real_data(self):
        fp = epm.reader.feature_parser.FeatureParser()
        # only 119 features available
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

        pcs_file = os.path.join(self.src_dir, "files", "pcs",
                                "clasp-sat-params-nat.pcs")
        cs = epm.pcs.config_space.ConfigSpace(pcs_file, normalize=False)

        config_matrix = numpy.zeros((len(config_list), cs._n_params),
                                    dtype=numpy.float32)
        for indx, config in enumerate(config_list):
            config_vec = cs.convert_param_dict(config)
            # mean imputation
            imputed_vec = cs.impute_non_active(config_vec, value="def")
            config_matrix[indx] = imputed_vec

        imp = epm.preprocessing.fast_imputor_missing_y.FastImputorY(debug=True)
        max_y = 300
        max_data_points = 300

        base = functools.partial(RandomForestRegressor, n_estimators=2,
                                 random_state=self.rng)
        model = functools.partial(epmbootstrap.bootstrapModel,
                                  rng=self.rng, debug=False,
                                  n_bootstrap_samples=5, bootstrap_size=0.7,
                                  base_estimator=base)

        imp.impute_y(numpy.array(perf_list)[:max_data_points],
                     numpy.array(success_list[:max_data_points]) == False,
                     cutoff=max_y,
                     configs_list=numpy.array(config_matrix)[:max_data_points],
                     inst_list=instance_list[:max_data_points],
                     inst_feature_dict=inst_feat_dict,
                     model=model, log=True)


if __name__ == "__main__":
    unittest.main()
