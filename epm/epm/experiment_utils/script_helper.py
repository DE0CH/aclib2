import copy
import logging

import numpy
from scipy import stats

import sklearn.metrics

from epm.experiment_utils.helper import get_feature_normalizer
from epm.experiment_utils.data_handling import separate_data_with_bools, \
    build_data
# Collection of methods that are helpful in more than one script

logger = logging.getLogger("script_helper")


def get_min_max(pred, real):
    min_val = min(pred[0])
    max_val = max(pred[0])
    for i in range(len(pred)):
        min_val = min(min(pred[i]), min_val)
        min_val = min(min(real[i][1]), min_val)
        max_val = max(max(pred[i]), max_val)
        max_val = max(max(real[i][1]), max_val)
    diff = abs(max_val-min_val)
    max_val += 0.1*diff
    if min_val-0.1*diff > 0:
        min_val -= 0.1*diff

    return min_val, max_val


def wrap_rmse(y_true, y_pred):
    return numpy.sqrt(sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred))


def wrap_spearman(y_true, y_pred):
    coef, _p = stats.spearmanr(a=y_pred, b=y_true)
    return coef


def new_preprocess_data(idx, encoded_mat, inst_list, perf_list, cen_list,
                        inst_feat_dict, n_feats_used, feature_normalizer=None):
    """
    Takes output from load data and preprocesses data to be ready to fit/predict
    :param idx: list
           list with indices to use from encoded_mat
    :param encoded_mat: array
           array with preprocessed configurations
    :param inst_list: list
           list with instance_keys, corresponds to inst_feat_dict
    :param perf_list: list
           list with y-values for encoded_mat
    :param cen_list: list
           list with booleans, True if datapoint is censored, False otherwise
    :param inst_feat_dict: dict
           dictionary with instance features
    :param n_feats_used: int
           number of used instance features
    :param feature_normalizer: feature_normalizer
           feature_normalizer instance
           if None a new feature_normalizer will be trained
    :return:
    """
    # Split data
    if cen_list is not None:
        cen_enc_mat, cen_inst_ls, cen_y, uncen_enc_mat, uncen_inst_ls, uncen_y = \
            separate_data_with_bools(data_matrix=copy.deepcopy(encoded_mat[idx, :]),
                                     inst_list=[inst_list[i] for i in idx],
                                     perf_list=[perf_list[i] for i in idx],
                                     succ_list=[cen_list[i] for i in idx])
    else:
        logger.info("No censored data found")
        uncen_enc_mat = encoded_mat
        uncen_inst_ls = inst_list
        uncen_y = perf_list
        cen_enc_mat, cen_inst_ls, cen_y = (None, None, None)

    # if features normalize is not given, we probably have training data
    if feature_normalizer is None:
        logger.info("# Create normalizer")
        feature_normalizer = get_feature_normalizer(
                train_lists=(cen_inst_ls, uncen_inst_ls),
                inst_feat_dict=inst_feat_dict)
    else:
        logger.info("# Reuse normalizer")

    inst_feat_dict = feature_normalizer.normalization(inst_feat_dict)

    logger.info("# Features are normalized")
    uncen_X = build_data(data_matrix=uncen_enc_mat,
                         inst_list=uncen_inst_ls,
                         inst_feat_dict=inst_feat_dict,
                         n_feats=n_feats_used)
    if cen_enc_mat is not None:
        cen_X = build_data(data_matrix=cen_enc_mat,
                           inst_list=cen_inst_ls,
                           inst_feat_dict=inst_feat_dict,
                           n_feats=n_feats_used)
    else:
        cen_X = None
    return feature_normalizer, uncen_X, uncen_y, cen_X, cen_y