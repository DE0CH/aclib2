"""
Created on October 07, 2015

@author: Marius Lindauer
"""

import logging

import numpy as np
import pickle
from pathlib import Path
from pyrfr import regression as regression
from epm.experiment_utils.config_space_utils import get_categorical_from_cs


class ExternalRFR(object):

    def __init__(self,
                 cs,
                 n_feats,
                 do_bootstrapping=True,
                 min_samples_to_split=2,
                 min_samples_in_leaf=1,
                 max_depth=0,
                 max_num_nodes=0,
                 epsilon_purity=1e-8,
                 num_trees=10,
                 seed=12345,
                 frac_points_per_tree=0,
                 ratio_features=0,
                 debug=False,
                 dtype=np.float32):
        """
        
        We assume that x is encoded as config||features

        :param cs: ConfigSpace.ConfigurationSpace
        :param n_feats: number of instance features (int)
        :param do_bootstrapping: bootstrapping? (bool)
        :param num_data_points_per_tree: number of data points per tree (int)
        :param max_features_per_split: maximal number of features per split (int)
        :param min_samples_to_split: minimal number of samples to split a node (int)
        :param min_samples_in_leaf: minimal number of samples in leafs after a split (int)
        :param max_depth:  maximal depth of a tree (int)
        :param epsilon_purity: purity of leafs (double)
        :param num_trees: number of trees (int)
        :param seed: random seed (int)
        :param debug: debug output (bool)
        :return: self
        """

        self.logger = logging.getLogger("externalRFR")
        self.debug = debug
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # These parameters cannot be set directly
        self.frac_points_per_tree = frac_points_per_tree
        self.ratio_features = ratio_features
        self.seed = seed

        # logging.info("Use %s as EPM model" % (name))
        model = regression.binary_rss_forest()
        model.options.do_bootstrapping = do_bootstrapping
        model.options.tree_opts.min_samples_to_split = min_samples_to_split
        model.options.tree_opts.min_samples_in_leaf = min_samples_in_leaf

        model.options.tree_opts.max_depth = max_depth
        model.options.tree_opts.max_num_nodes = max_num_nodes
        model.options.tree_opts.epsilon_purity = epsilon_purity
        model.options.num_trees = num_trees
        self.model = model

        types_params = np.array(get_categorical_from_cs(cs=cs))
        types_feats = np.zeros(n_feats, dtype=np.uint64)
        self.logger.debug("Types params shape: %s" % (str(types_params.shape)))
        self.logger.debug("Types feats shape: %s" % (str(types_feats.shape)))
        types = np.hstack((types_params, types_feats))
        self.logger.debug("Types shape: %s" % (str(types.shape)))
        self.logger.debug("Types: %s" % (str(types)))

        self.types = np.array(types, dtype=np.uint64)

        self._dtype = dtype
        self.model_type = 'binary_rss_forest'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = logging.getLogger("externalRFR_resurrected")

        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def fit(self, X, y):
        """
        Trains a random forest regression on X
        :param X: np array mxn, required, features
        :param y: np array m, required, objective value
        :return: self
        """

        data = regression.default_data_container(len(X[0]))
        [data.add_data_point(row_X.tolist(), row_y.item())
            for row_X, row_y in zip(X, y)]

        self.model.options.num_data_points_per_tree = \
            max(0, int(X.shape[0] * self.frac_points_per_tree))
        self.model.options.tree_opts.max_features = \
            max(0, int(X.shape[1] * self.ratio_features))

        self.model.fit(data, regression.default_random_engine(self.seed))

        # data1 = regression.np_data_container(X, y, self.types)
        # self.model.num_data_points_per_tree = \
        #     max(0, int(X.shape[0]*self.frac_points_per_tree))
        # self.model.max_features = max(0, int(X.shape[1]*self.ratio_features))
        # self.model.fit(data1)
        return self

    def train(self, X, y):
        # TODO: (pm, 12.5) REWRITE train/fit method
        return self.fit(X, y)

    def change_type_array_to_outlier_imputation(self):
        self.logger.critical("Change type array to handle 'outlier' "
                             "feature imputation")
        self.types = np.array([int(i+1) if i != 0 else i for i in self.types],
                                 dtype=np.uint64)

    def predict_mean_std(self, X):
        """
        returns mean/var prediction
        :param x: np vector/matrix, required, features
        :return: mean, variance 
        """
        mean, var = self.predict_mean_var(X)
        std = np.sqrt(var)

        return mean, std

    def predict_mean_var(self, X):
        """
        returns mean/sqrt(var) prediction
        :param x: np vector/matrix, required, features
        :return: mean, variance 
        """
        if self.model is None:
            raise ValueError("Model is not yet trained")

        threshold = 10 ** -10
        if len(X.shape) > 1:
            pred = np.array([self.model.predict_mean_var(x.tolist())
                                for x in X])
            mean = pred[:, 0]
            var = pred[:, 1]
            var[var < threshold] = threshold
            var[np.isnan(var)] = threshold

            mean = np.array(mean, dtype=self._dtype)
            var = np.array(var, dtype=self._dtype)
        else:
            mean, var = self.model.predict_mean_var(X)
            if var < threshold:
                self.logger.debug("Standard deviation is %f, capping to 10^-5" % var)
                var = threshold
            var = np.array([var, ], dtype=self._dtype)
            mean = np.array([mean, ], dtype=self._dtype)

        mean = mean.reshape((-1, 1))
        var = var.reshape((-1, 1))

        return mean, var

    def predict(self, X):
        """
        exists to be compatible to sklearn estimators
        :param x: np vector/matrix, required, features
        :return: mean prediction
        """
        # TODO (pm, 13.5): Return Variance, to make it work with SMAC imputor
        return self.predict_mean_var(X=X)
        # return self.predict_mean_std(X=X)[0]

    def out_of_bag_error(self):
        return self.model.out_of_bag_error()


def save_model(wrapper, save_path, model_name=None, wrapper_name=None):
    """
    Save the model to file. Since there is a problem with pickling the pyrfr
    directly, this method performs the serializing more carefully.
    Save the internal pyrfr (SWIG object) to binary using its internal save
    method. The wrapper function, is stored with pickle.

    Note: The problem 'Model cant be pickled' occurs if the model was
          trained on too many datapoints. Getting the ascii representation
          of the model causes a memory overflow. (more than 16GB RAM by
          3.2Gb data)

    Parameters
    ----------
    wrapper : external_rfr object
        the wrapper for the swig object
    save_path : str, Pathlike-object
        directory, where data should be stored. Must exist!
    model_name : str, optional
        file name of the model. If None is specified a default name is used
    wrapper_name
        file name of the wrapper. If None is specified a default name is
         used
    """

    pyrfr_path, wrapper_path = \
        get_output_paths(save_path, model_name, wrapper_name)

    if pyrfr_path.exists():
        wrapper.logger.warning('A stored model already exists and will be '
                         'overwritten')

    wrapper.model.save_to_binary_file(str(pyrfr_path))

    # Store Wrapper with pickle. First, delete stored pyrfr model.
    if wrapper_path.exists():
        wrapper.logger.warning('A stored wrapper already exists and will be '
                         'overwritten')

    wrapper.model = None

    pickle.dump(obj=wrapper,
                file=open(str(wrapper_path), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    wrapper.logger.debug('Model successfully stored to {}'.format(save_path))


def load_model(save_path, model_name=None, wrapper_name=None):
    """
    Load the model from file. All files must lie in the directory save_path!

    Parameters
    ----------
    wrapper : external_rfr object
        the wrapper for the swig object
    save_path : str, Pathlike-object
        directory, from where the data should be loaded. Must exist!
    model_name : str, optional
        file name of the model. If None is specified a default name is used
    wrapper_name
        file name of the wrapper. If None is specified a default name is
         used
    """
    logger = logging.getLogger("externalRFR_resurrected")

    pyrfr_path, wrapper_path = \
        get_output_paths(save_path, model_name, wrapper_name)

    # Store Wrapper with pickle. First, delete stored pyrfr model.
    if not wrapper_path.exists():
        msg = 'Can\'t find stored pyrfr wrapper in {}'.format(wrapper_path)
        logger.warning(msg)
        raise FileNotFoundError(msg)

    wrapper = pickle.load(open(str(wrapper_path), 'rb'))

    if not pyrfr_path.exists():
        msg = 'Can\'t find stored pyrfr model in {}'.format(pyrfr_path)
        logger.warning(msg)
        raise FileNotFoundError(msg)

    try:
        if wrapper.model_type == 'binary_rss_forest':
            model = regression.binary_rss_forest()
        elif wrapper.model_type == 'qr_forest':
            model = regression.qr_forest()
        else:
            msg = 'Modeltype {} unknown. Should be one of [qr_forest,' \
                  ' binary_rss_forest'.format(wrapper.model_type)
            logger.error(msg)
            raise ValueError(msg)
    except AttributeError as e:
        logger.exception('Modeltype is not specified. Therefore, the model was '
                         'trained with an old training script. Please retrain, '
                         'with new epm module.')
        raise e
    except Exception as e:
        logger.exception('Some errror occured in pyrfr!')
        raise e

    model.load_from_binary_file(str(pyrfr_path))
    wrapper.model = model
    logger.debug('RandomForrest successfully loaded from {}'.format(save_path))
    return wrapper


def get_output_paths(save_path, model_name, wrapper_name):
    """ Helper func to determine paths, where stored data is saved. """
    save_path = Path(save_path).absolute()

    # Since model and wrapper_name can be paths to the files, take only the file
    # name and strip file endings.
    model_name = Path(model_name).name.rstrip('.bin').rstrip('.pkl')
    wrapper_name = Path(wrapper_name).name.rstrip('.bin').rstrip('.pkl')

    # make sure output directory exists:
    if not save_path.exists():
        raise FileNotFoundError('Output directory does not exist: {}'
                                .format(save_path))

    pyrfr_path = model_name or 'pyrfr_model'
    pyrfr_path += '.bin' if not pyrfr_path.endswith('.bin') \
        else ''
    pyrfr_path = save_path / pyrfr_path

    wrapper_path = wrapper_name or 'pyrfr_wrapper'
    wrapper_path += '.pkl' if not wrapper_path.endswith('.pkl') \
        else ''
    wrapper_path = save_path / wrapper_path

    return pyrfr_path, wrapper_path
