import numpy as np
import logging
import ConfigSpace as CS
from ConfigSpace.hyperparameters import NumericalHyperparameter, \
    CategoricalHyperparameter
from ConfigSpace.util import fix_types, impute_inactive_values
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger()


def encode_config_as_array_with_true_values(config, cs, normalize=False):
    """
    Method to produce same array representation as the old configspace module
    from the aclib. It is similar to the array representation of the
    configspace module, but continuous hyperparameter are not scaled between
    0 and 1, but instead replaced by their "true" value.

    Note: In the ConfigSpace package, categorical hyperparameter are replaced by
          the indices of the value in its choices.

    Parameters
    ----------
    config : configuration
    cs : ConfigSpace object
    normalize : bool, optional
        If True, all values are scaled between 0 and 1. categorical values are
        simply the index of their values in the hyperparameters choices.

    Returns
    -------
    np.array
        Array Representation of the configuration as describe above
    """
    cfg_array = config.get_array()

    if normalize:
        return cfg_array

    # This is not necessary for inactive parameter, because for they already
    # default value is used
    active_hp_names = [name
                       for name in cs.get_active_hyperparameters(config)]
    inactive_hp_names = set(cs.get_hyperparameter_names()) \
                        - set(active_hp_names)

    for i, hp in enumerate(cs.get_hyperparameters()):
        if hp.name in inactive_hp_names:
            continue

        if issubclass(type(hp), NumericalHyperparameter):
            value = config[hp.name]
            value = np.log(value) if hp.log else value
            cfg_array[i] = value

    return cfg_array


def one_hot_encode(config, cs):
    """
    Encodes a configuration from a configspace cs in
    one hot-representation.

    Parameters
    ----------
    config : ConfigSpace.configuration
        The configuration with D parameters to one-hot encode
    cs : ConfigSpace.configuration_space
        The configuration space from which the configuration is sampled
    Returns
    -------
    np.array
        one hot encoded array representation of the configuration. Shape: 1xD'
        with D > D'
    """
    values = [np.sort(np.array(hp.choices))
              if type(hp) is CategoricalHyperparameter else []
              for hp in cs.get_hyperparameters()]

    transformer = [OneHotEncoder(categories='auto').fit(v.reshape((-1, 1)))
                   if len(v) is not 0 else None
                   for v in values]

    # Create array with one hot encoded values
    # config_vec = config.get_array().reshape((-1, 1))
    result_vec = np.array([]).reshape((-1, 1))

    for i, name in enumerate(cs.get_hyperparameter_names()):
        val = np.array(config.get(name)).reshape((-1, 1))
        if transformer[i] is None:
            result_vec = np.hstack((result_vec, val)) \
                if len(result_vec) != 0 else val

        else:
            transformed = transformer[i].transform(val).toarray()
            result_vec = np.hstack((result_vec, transformed)) \
                if len(result_vec) != 0 else transformed

    return result_vec


def one_hot_encode_array(X, cat_list=None):
    """
    Perform a one-hot-encoding on a given numpy array.

    Parameters
    ----------
    X  : np.array
        Dimension: [n samples, n features]

    cat_list
        List representing for each column the number of choices for a
        categorical hyperparameter.
        If a column does not represent a categorical, cat_list is 0.

    Returns
    -------
        One Hot encoded matrix.
    """
    assert isinstance(X, np.ndarray)
    assert X.shape[1] == len(cat_list)

    # Not sure whether this is necessary
    tmpx = X

    col_idx = 0
    for idx, entry in enumerate(cat_list):
        if entry == 0:
            col_idx += 1
            continue
        elif entry == 1:
            # logging.debug("%d has only one categorical value" % idx)
            # Therefore all entries must be 0
            assert (tmpx[:, col_idx] == 0).all(), "Not all entries are 0"
            col_idx += 1
        else:
            # extract and remove to-encode-row
            seg_col = tmpx[:, col_idx]

            # make sure this is a categorical row
            assert (np.array(seg_col, dtype=int) == seg_col).all()

            tmpx = np.hstack((tmpx[:, :col_idx], tmpx[:, col_idx + 1:]))

            for i in range(entry):
                # Compare array with each possible value and insert 'bool' array
                insert = np.reshape((seg_col == i).astype(float),
                                    (tmpx.shape[0], 1))
                tmpx = np.hstack(
                        (tmpx[:, :col_idx], insert, tmpx[:, col_idx:]))
                col_idx += 1
    return tmpx


def get_imputed_config_from_dict(config, cs, impute_with='default'):
    """
    Create a configuration from a dictionary. A configuration created with the
    old configuration space module (2015 from aclib),
    may contain inactive hyperparameters. Therefore, some preprocessing is
    necessary.

    Parameters
    ----------
    config : dict
        dictionary representation of a ConfigSpace.Configuration
    cs : ConfigSpace.ConfigurationSpace

    impute_with : str, optional
        imputation strategy. Defaults to 'def'

    Returns
    -------
    ConfigSpace.Configuration
    """

    config_dict = get_imputed_config_as_dict_from_dict(config, cs, impute_with)

    # Allow temporarily inactive values to allow the building of the
    # configuration object. Otherwise a error will be raised.
    # But after creating the configuration, the inactive parameters will be
    # imputed (similar to old config space).
    config = CS.Configuration(configuration_space=cs,
                              values=config_dict,
                              allow_inactive_with_values=True)

    # TODO: These steps are unnecessary! (exact same behaviour like above, but
    #       for dictionaries.
    # make sure it works with new configspace ('def'--> 'default')
    impute_with = 'default' if impute_with == 'def' else impute_with
    config = impute_inactive_values(config, impute_with)

    return config


def get_imputed_config_as_dict_from_dict(config, cs, impute_with='def'):
    """
    Create a configuration in dictionary representation from a dictionary.
    A configuration created with the old configuration space module
    (2015 from aclib), may contain inactive hyperparameters.
    Therefore, some preprocessing is necessary.

    Parameters
    ----------
    config : dict
        dictionary representation of a ConfigSpace.Configuration
    cs : ConfigSpace.ConfigurationSpace

    impute_with : str,int,float
        imputation strategy. Defaults to 'def'

    Returns
    -------
    ConfigSpace.Configuration
    """

    config_dict = fix_types(config, cs)
    # include missing (inactive parameters)
    if impute_with == 'def':
        config_dict = \
            {name:
             config_dict.get(name, cs.get_hyperparameter(name).default_value)
             for name in cs.get_hyperparameter_names()}
    if type(impute_with) in [int, float, np.float, np.int]:
        config_dict = \
            {name:
             config_dict.get(name, impute_with)
             for name in cs.get_hyperparameter_names()}

    return config_dict


def get_categorical_from_cs(cs):
    """
    Create a list with the number of categorical choices per hyperparameter of a
    ConfigSpace. If hp is not a categorical hyperparameter, a 0 is added.

    Parameters
    ----------
    cs : ConfigSpace.ConfigurationSpace

    Returns
    -------
    list
    """

    cat_size = []
    for hp in cs.get_hyperparameters():
        try:
            cat_size.append(len(hp.choices))
        except AttributeError:
            cat_size.append(0)
    return cat_size
