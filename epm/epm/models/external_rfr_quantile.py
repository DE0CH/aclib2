"""
Created on April 05, 2016

@author: Katharina Eggensperger
"""

import numpy

from epm.models.external_rfr import ExternalRFR
from pyrfr import regression


class ExternalRFRQuantile(ExternalRFR):

    def __init__(self, cs, n_feats,
                 do_bootstrapping=True, min_samples_to_split=2,
                 min_samples_in_leaf=1, max_depth=0, epsilon_purity=1e-8,
                 num_trees=10, seed=12345, debug=False, dtype=numpy.float32,
                 max_num_nodes=0, frac_points_per_tree=0, ratio_features=0
                 ):
        super(ExternalRFRQuantile, self).__init__(
                cs=cs, n_feats=n_feats, do_bootstrapping=do_bootstrapping,
                ratio_features=ratio_features,
                min_samples_to_split=min_samples_to_split,
                min_samples_in_leaf=min_samples_in_leaf, max_depth=max_depth,
                epsilon_purity=epsilon_purity, num_trees=num_trees,
                max_num_nodes=max_num_nodes,
                frac_points_per_tree=frac_points_per_tree, seed=seed,
                debug=debug, dtype=dtype)

        model = regression.qr_forest()
        model.options.do_bootstrapping = do_bootstrapping
        model.options.tree_opts.min_samples_to_split = min_samples_to_split
        model.options.tree_opts.min_samples_in_leaf = min_samples_in_leaf

        model.options.tree_opts.max_depth = max_depth
        model.options.tree_opts.max_num_nodes = max_num_nodes
        model.options.tree_opts.epsilon_purity = epsilon_purity
        model.options.num_trees = num_trees

        self.model = model
        self.model_type = 'qr_forest'
        pass

    def predict_mean_std(self, X):
        raise NotImplementedError("Quantile Regression tree cannot return std")

    def predict_mean_var(self, X):
        raise NotImplementedError("Quantile Regression tree cannot return var")

    def _predict_quantile_direct(self, X, quantile):
        if len(X.shape) > 1:
            pred = ([self.model.quantile_prediction(
                    x, numpy.array([quantile, ], dtype=self._dtype))
                     for x in X])
        else:
            pred = self.model.quantile_prediction(
                    X, numpy.array([quantile, ], dtype=self._dtype))

        pred = numpy.array(pred, dtype=self._dtype)
        pred = pred.flatten()
        return pred

    def _predict_quantile(self, x, seed, num_samples=1):
        if seed == 0:
            self.logger.debug("Return 0.5 quantile (~median)")
            return self.model.predict_quantiles(x.tolist(), [0.5])
        else:
            rs = numpy.random.RandomState(seed)
            alphas = numpy.array(rs.uniform(low=0, high=1, size=num_samples),
                                 dtype=self._dtype)
            return self.model.predict_quantiles(x.tolist(), alphas.tolist())

    def predict(self, X, seed, num_samples=1):
        """
        exists to be compatible to sklearn estimators
        :param X: numpy vector/matrix, required
               input features
        :param seed: int
               if 0 or 2147483647 return median and ignore num_samples
               else use this to seed randomState
        :return: prediction
        """
        if num_samples < 1:
            self.logger.critical("I cannot draw a negative number of samples, "
                                 "so I will only draw 1")
            num_samples = 1

        if seed == 0 or seed == 2147483647 or seed == -1:
            self.logger.critical("Seed in (0, -1, 2147483647), "
                                 "so I return median")
            seed = 0

        if seed == 0 and num_samples > 1:
            self.logger.critical("Seed in (0, -1, 2147483647), "
                                 "so I draw only 1 sample")
            num_samples = 1

        if len(X.shape) > 1:
            pred = ([self._predict_quantile(x=x, seed=seed,
                                            num_samples=num_samples)
                     for x in X])
        else:
            pred = self._predict_quantile(X, seed=seed, num_samples=num_samples)
        pred = numpy.array(pred, dtype=self._dtype)

        if num_samples == 1:
            pred = pred.flatten()

        return pred
