"""
Created on Mar 27, 2015

@author: Katharina Eggensperger
"""

import logging
import sys

import numpy

from sklearn.utils import resample

from epm import MAXINT


class bootstrapModel(object):

    def __init__(self, rng, base_estimator, n_bootstrap_samples,
                 bootstrap_size=0.7, debug=False, max_value=MAXINT):
        """

        :param rng: int, required, random number used to initialize bootstrapping
        :param base_estimator: python object, required, sklearn class that implements fit and predict
                               with predefined arguments
        :param n_bootstrap_samples: int, required, number of boostraps ~ number of used models
        :param bootstrap_size: [0.0, 1.0), default 0.7,  represent the size of the training
                               dataset for each model wrt to the original size
        :param debug: boolean, default False not used so far
        :return: self
        """
        self.models = list()
        self.rng = rng
        self.debug = debug
        self.base_estimator = base_estimator
        self.n_bootstrap_samples = n_bootstrap_samples
        self.bootstrap_size = bootstrap_size
        self.max_value = max_value
        self.rs = numpy.random.RandomState(rng)
        self._dtype = numpy.float32

        self.logger = logging.getLogger("bootstrapModel")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)

        assert 0.0 <= self.bootstrap_size <= 1.0

    def fit(self, X, y):
        """
        Trains n_bootstrap_samples models on subsamples of X
        :param X: numpy array mxn, required, features
        :param y: numpy array m, required, objective value
        :return: self
        """

        # Might be better to do some checks
        assert isinstance(X, numpy.ndarray)
        assert isinstance(y, numpy.ndarray)

        if X.dtype != numpy.float32:
            raise ValueError("X has dtype %s, expected %s" %
                             (X.dtype, self._dtype))
        if y.dtype != numpy.float32:
            raise ValueError("y has dtype %s, expected %s" %
                             (y.dtype, self._dtype))
        assert X.shape[0] == len(y)

        if not numpy.isfinite(X).all():
            self.logger.critical("X array is not finite")
            raise ValueError("X array is not finite")
        if not numpy.isfinite(y).all():
            self.logger.critical("y array is not finite")
            raise ValueError("y array is not finite")

        if self.debug:
            try:
                # Often we have a 'partial' function
                self.logger.debug("Base estimator: %s" % self.base_estimator.func)
            except:
                self.logger.debug("Base estimator: %s" % str(self.base_estimator))

            self.logger.debug("Train X.size: %s" % str(X.shape))
            _min = numpy.min(X, axis=0)
            min_less_0 = _min < 0
            self.logger.debug("Train X.min < 0: %s" % str(min_less_0.any()))
            if min_less_0.any():
                self.logger.error("Found min > 0: %f" % numpy.min(X))

            _max = numpy.max(X, axis=0)
            max_more_1 = _max > 1
            self.logger.debug("Train X.max > 1: %s" % str(max_more_1.any()))
            if max_more_1.any():
                self.logger.error("Found max > 1: %f" % numpy.max(X))

            self.logger.debug("Train X Mean: %s" % str(numpy.mean(X)))

        # define size for each subset sample
        sample_size = int(X.shape[0]*self.bootstrap_size)
        assert sample_size > 1

        # fit n_bootstrap_samples models, increase seed by 1 for each iteration
        for n in range(self.n_bootstrap_samples):
            self.logger.debug("Training model %d" % n)
            X_train, y_train = resample(X, y, replace=True, n_samples=sample_size, random_state=self.rng+n)
            tmp_model = self.base_estimator()
            tmp_model.fit(X_train, y_train)
            self.models.append(tmp_model)

        return self

    def _check_before_predict(self, X):
        if self.models is None:
            raise ValueError("Model is not yet trained")

        assert type(X) == numpy.ndarray

        if X.dtype != numpy.float32:
            raise ValueError("X has dtype %s, expected float32" % str(X.dtype))

        assert X.ndim == 2

    def _prepare_output(self, X):
        # We don't want anything larger than max_value
        X = [sys.maxsize if x > self.max_value else x for x in X]

        X = numpy.array(X, dtype=numpy.float32)
        if len(X.shape) == 1:
            return X
        else:
            self.logger.critical("Found a array with shape %s, return flattened" % str(X.shape))
            return X.flatten()

    def predict_mean_std(self, X):
        """
        Iterates over all models and returns mean and std
        :param X: numpy array mxn, required, features
        :return: (mean, std)
        """
        self._check_before_predict(X)

        if self.debug:
            self.logger.debug("Predict X.size: %s" % str(X.shape))
            _min = numpy.min(X, axis=0)
            min_less_0 = _min < 0
            self.logger.debug("Predict X.min < 0: %s" % str(min_less_0.any()))
            if min_less_0.any():
                self.logger.error("Found min < 0: %s" % numpy.min(X))

            _max = numpy.max(X, axis=0)
            max_more_1 = _max > 1
            self.logger.debug("Predict X.max > 1: %s" % str(max_more_1.any()))
            if max_more_1.any():
                self.logger.error("Found max > 1: %f" % numpy.max(X))

            self.logger.debug("Predict X Mean: %s" % str(numpy.mean(X)))

        preds = [e.predict(X) for e in self.models]
        mean = self._prepare_output(numpy.mean(preds, axis=0))
        std = self._prepare_output(numpy.std(preds, axis=0))

        return mean, std

    def normal_sample_predict(self, X):
        """
        Calls predict and samples from a normal distribution
        :param X:
        :return: pred
        """
        self._check_before_predict(X)

        mean, std = self.predict_mean_std(X)
        # Yippieh
        pred = map(lambda m, s: self.rs.normal(m, s), mean, std)

        pred = self._prepare_output(pred)

        return pred
