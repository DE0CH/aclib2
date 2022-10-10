"""
Created on October 07, 2015

@author: Marius Lindauer
"""

import numpy
import scipy.stats

from epm.models.external_rfr import ExternalRFR
from epm.experiment_utils.data_handling import warp, unwarp


class ExternalRFRPredict(ExternalRFR):

    def __init__(self, cs, n_feats, cutoff, threshold, prediction_threshold=0,
                 do_bootstrapping=True, min_samples_to_split=2,
                 min_samples_in_leaf=1, max_depth=0, epsilon_purity=1e-8,
                 num_trees=10, seed=12345, debug=False, dtype=numpy.float32,
                 max_num_nodes=0, frac_points_per_tree=0, ratio_features=0,
                 ):
        super(ExternalRFRPredict, self).__init__(
                cs=cs, n_feats=n_feats, do_bootstrapping=do_bootstrapping,
                ratio_features=ratio_features,
                min_samples_to_split=min_samples_to_split,
                min_samples_in_leaf=min_samples_in_leaf, max_depth=max_depth,
                epsilon_purity=epsilon_purity, num_trees=num_trees,
                max_num_nodes=max_num_nodes,
                frac_points_per_tree=frac_points_per_tree, seed=seed,
                debug=debug, dtype=dtype)
        self.cutoff = cutoff
        self.threshold = threshold
        self.prediction_threshold = prediction_threshold

    def predict(self, X, cutoff=None, threshold=None, prediction_threshold=None,
                logged=False):
        """
        exists to be compatible to sklearn estimators
        :param X: numpy vector/matrix, required
               input features
        :param cutoff : float, optional
               cutoff value from ACLib scenario.
               If None, initial cutoff will be used
        :param threshold : float, optional
               there will be no return value higher than this, e.g. cutoff*par
               If None, initial cutoff will be used
        :param prediction_threshold: float, optional
               is less mass then this > cutoff, return mean prediction
               If None, initial cutoff will be used
        :param logged : bool, optional
               if trained on log10-data, method will use war/unwarp to predict
        :return: prediction
        """
        if cutoff is None:
            tmp_cutoff = self.cutoff
        else:
            tmp_cutoff = cutoff

        if threshold is None:
            tmp_threshold = self.threshold
        else:
            tmp_threshold = threshold

        if prediction_threshold is None:
            tmp_pt = self.prediction_threshold
        else:
            tmp_pt = prediction_threshold

        mean_std = self.predict_mean_std(X=X)

        pred = numpy.zeros(shape=mean_std[0].shape, dtype=self._dtype)
        for p in range(pred.shape[0]):
            if mean_std[0][p] > tmp_cutoff:
                # mean prediction is already higher than cutoff
                self.logger.critical("Predicted %g which is higher than cutoff"
                                     " %s" % (mean_std[0][p], tmp_cutoff))
                #pred[p] = tmp_threshold
                #continue

            # Calc cdf from -inf to cutoff
            cdf = scipy.stats.norm.cdf(x=tmp_cutoff, loc=mean_std[0][p],
                                       scale=mean_std[1][p])

            # Probability mass > cutoff
            upper_exp = 1-cdf

            if upper_exp > 1:
                self.logger.warn("Upper exp is larger than 1, "
                                     "is this possible: %g > 1" % upper_exp)
                upper_exp = 1
                cdf = 0

            if upper_exp < tmp_pt or tmp_threshold == tmp_cutoff:
                # There is not enough probability mass higher than cutoff
                # Or threshold == cutoff
                pred[p] = mean_std[0][p]
            else:
                # Calculate mean of lower truncnorm
                lower_pred = scipy.stats.truncnorm.stats(
                        a=(-numpy.inf - mean_std[0][p]) / mean_std[1][p],
                        b=(tmp_cutoff - mean_std[0][p]) / mean_std[1][p],
                        loc=mean_std[0][p],
                        scale=mean_std[1][p],
                        moments='m')
                # Add truncnorm mean*cdf + threshold*cdf
                if logged:
                    # We have to use unwarped values as '+' in logspace is not
                    # equal to + in non-logspace
                    upper_pred = upper_exp * unwarp(tmp_threshold, quality=False)
                    lower_pred = cdf * unwarp(lower_pred, quality=False)
                    pred[p] = warp(lower_pred+upper_pred, quality=False)
                else:
                    upper_pred = upper_exp * tmp_threshold
                    pred[p] = lower_pred * cdf + upper_pred

                if pred[p] > tmp_threshold + 10**-5:
                    raise ValueError("Predicted higher than possible, %g > %g"
                                     % (pred[p], tmp_threshold))

                # This can happen and if it happens, set prediction to cutoff
                if not numpy.isfinite(pred[p]):
                    self.logger.critical("Prediction is not finite cdf %g, "
                                         "lower_pred %g; Setting %g to %g" %
                                         (cdf, lower_pred, pred[p],
                                          tmp_cutoff + 10**-5))
                    pred[p] = tmp_cutoff + 10**-5
        return pred

    def predict_mv(self, X, cutoff=None, threshold=None, prediction_threshold=None,
                logged=False):
        """
        exists to be compatible to sklearn estimators
        :param X: numpy vector/matrix, required
               input features
        :param cutoff : float, optional
               cutoff value from ACLib scenario.
               If None, initial cutoff will be used
        :param threshold : float, optional
               there will be no return value higher than this, e.g. cutoff*par
               If None, initial cutoff will be used
        :param prediction_threshold: float, optional
               is less mass then this > cutoff, return mean prediction
               If None, initial cutoff will be used
        :param logged : bool, optional
               if trained on log10-data, method will use war/unwarp to predict
        :return: prediction
        """
        if cutoff is None:
            tmp_cutoff = self.cutoff
        else:
            tmp_cutoff = cutoff

        if threshold is None:
            tmp_threshold = self.threshold
        else:
            tmp_threshold = threshold

        if prediction_threshold is None:
            tmp_pt = self.prediction_threshold
        else:
            tmp_pt = prediction_threshold

        mean_std = self.predict_mean_std(X=X)

        pred = numpy.zeros(shape=mean_std[0].shape, dtype=self._dtype)
        var = numpy.zeros(shape=mean_std[0].shape, dtype=self._dtype)
        for p in range(pred.shape[0]):
            var[p] = mean_std[1][p]**2
            if mean_std[0][p] > tmp_cutoff:
                # mean prediction is already higher than cutoff
                self.logger.critical("Predicted %g which is higher than cutoff"
                                     " %s" % (mean_std[0][p], tmp_cutoff))
                #pred[p] = tmp_threshold
                #continue

            # Calc cdf from -inf to cutoff
            cdf = scipy.stats.norm.cdf(x=tmp_cutoff, loc=mean_std[0][p],
                                       scale=mean_std[1][p])

            # Probability mass > cutoff
            upper_exp = 1-cdf

            if upper_exp > 1:
                self.logger.warn("Upper exp is larger than 1, "
                                     "is this possible: %g > 1" % upper_exp)
                upper_exp = 1
                cdf = 0

            if upper_exp < tmp_pt or tmp_threshold == tmp_cutoff:
                # There is not enough probability mass higher than cutoff
                # Or threshold == cutoff
                pred[p] = mean_std[0][p]
            else:
                # Calculate mean of lower truncnorm
                lower_pred = scipy.stats.truncnorm.stats(
                        a=(-numpy.inf - mean_std[0][p]) / mean_std[1][p],
                        b=(tmp_cutoff - mean_std[0][p]) / mean_std[1][p],
                        loc=mean_std[0][p],
                        scale=mean_std[1][p],
                        moments='m')
                # Add truncnorm mean*cdf + threshold*cdf
                if logged:
                    # We have to use unwarped values as '+' in logspace is not
                    # equal to + in non-logspace
                    upper_pred = upper_exp * unwarp(tmp_threshold)
                    lower_pred = cdf * unwarp(lower_pred)
                    pred[p] = warp(lower_pred+upper_pred)
                else:
                    upper_pred = upper_exp * tmp_threshold
                    pred[p] = lower_pred * cdf + upper_pred

                if pred[p] > tmp_threshold + 10**-5:
                    raise ValueError("Predicted higher than possible, %g > %g"
                                     % (pred[p], tmp_threshold))

                # This can happen and if it happens, set prediction to cutoff
                if not numpy.isfinite(pred[p]):
                    self.logger.critical("Prediction is not finite cdf %g, "
                                         "lower_pred %g; Setting %g to %g" %
                                         (cdf, lower_pred, pred[p],
                                          tmp_cutoff + 10**-5))
                    pred[p] = tmp_cutoff + 10**-5
        return pred, var

