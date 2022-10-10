#!/usr/local/bin/python2.7
# encoding: utf-8
"""
EPM -- empirical performance models

@author:     Katharina Eggensperger and Marius Lindauer

@copyright:  2015 AAD Group Freiburg. All rights reserved.

@license:    GPLv2

@contact:    {eggenspk,lindauer}@cs.uni-freiburg.de
"""

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import copy
import functools
import logging
import os
import time

import numpy

from epm.experiment_utils.data_handling import load_data, partition_data, warp,\
    unwarp
from epm.preprocessing.fast_imputor_missing_y import FastImputorY as ImputorY
from epm.models import model_dict
from epm.pcs.config_space import ConfigSpace
from epm.experiment_utils.script_helper import new_preprocess_data, wrap_rmse

logger = logging.getLogger("evaluateRegression_ni_nc_PAR10")
model_debug = False
DOWNSAMPLE = 200000


def train_model(train_X, train_y, model):
    logger.info("Train shape: %s, %s" % (str(train_X.shape),
                                         str(train_y.shape)))
    start = time.time()
    m = model()
    m.fit(train_X, train_y)
    dur = time.time() - start
    logger.info("Training model took %s" % dur)

    return dur, m


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('target_algo_runs', metavar='JSON', nargs='+',
                        help='JSON files to read')
    parser.add_argument('--pcs', dest='pcs_file', default=None, required=True,
                        help='PCS file to read from')
    parser.add_argument('--features', dest='feature_file', required=False,
                        help='Feature file to read from')
    parser.add_argument('--instances', dest='instances', default=None,
                        help='Use instances given in that file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=sorted(model_dict.model_dict.keys()),
                        help="Which model to use?")
    parser.add_argument('--cutoff', dest='cutoff', default=-1,
                        required=False, type=float,
                        help="What was the cutoff time?")
    parser.add_argument('--save', dest='save', default=None,
                        help="Where to save predictions")
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help="Seed to use where applicable")
    parser.add_argument('--par', dest='par', default=1, type=int,
                        help="Treat TIMEOUTS as PARX")
    parser.add_argument('--quality', dest="quality", default=False,
                        action="store_true", help="Handle quality data?")

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(levelname)s][%(module)s]'
                               '  %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    logger.info("QUALITY: %s" % args.quality)

    rs = numpy.random.RandomState(args.seed)

    if not args.cutoff and not args.quality:
        raise ValueError("When using runtime, '--cutoff_ is required")

    # Tests whether file exists
    for f in (args.pcs_file, ):
        if not os.path.isfile(f):
            raise ValueError("%s is not a file" % f)

    if not os.path.isdir(os.path.dirname(args.save)):
        raise ValueError("%s is not a directory" % args.save)

    if args.instances is not None:
        if not os.path.isfile(args.instances):
            raise ValueError("%s is not a file" % args.instances)

    # Read in all data. No need to do oneHot, because data will be reread anyway
    enc_mat, inst_feat_dict, inst_ls, n_feats_used, perf_ls, to_ls, cen_ls = \
        load_data(args.target_algo_runs, feature_file=args.feature_file,
                  pcs_file=args.pcs_file, instance_file=args.instances,
                  normalize_features=False, cutoff=args.cutoff,
                  oneHot=False, par=args.par, downsample=DOWNSAMPLE,
                  quality=args.quality, normalize_parameters=False)

    if not args.quality:
        logger.info("%d censored data found" % sum(cen_ls))
        logger.info("%d TIMEOUT data found" % sum(to_ls))

    # Split data in four parts
    indices = partition_data(data_matrix=enc_mat, inst_list=inst_ls, rs=rs)
    logger.info("************ Partitions:")
    for c_idx in (0, 1):
            for i_idx in (0, 1):
                logger.info("%s,%s: %d" % (c_idx, i_idx,
                                           len(indices[c_idx][i_idx])))
    logger.info("************")

    # Now use diagonal pairs of datasets
    test_uncen_ls = dict()
    uncen_pred_ls = dict()
    uncen_trn_time = dict()
    imp_pred_ls = dict()
    imp_trn_time = dict()
    change_traj = dict()
    data_traj = dict()

    name, model, boot_model = model_dict.model_dict[args.model]

    # Special treatment for rfr
    if name == "rfr":
        cs = ConfigSpace(pcs_file=args.pcs_file, normalize=False)
        boot_model = functools.partial(boot_model.func, cs=cs,
                                       n_feats=n_feats_used,
                                       **boot_model.keywords)
        model = functools.partial(model.func, cs=cs, n_feats=n_feats_used,
                                  cutoff=warp(args.cutoff,
                                              quality=args.quality),
                                  threshold=warp(args.cutoff*2,
                                                 quality=args.quality),
                                  prediction_threshold=0, **model.keywords)
    elif name == "rfrq":
        cs = ConfigSpace(pcs_file=args.pcs_file, normalize=False)
        boot_model = functools.partial(boot_model.func, cs=cs,
                                       n_feats=n_feats_used,
                                       **boot_model.keywords)
        model = functools.partial(model.func, cs=cs, n_feats=n_feats_used,
                                  **model.keywords)

    data_hash = None
    logger.info("###############################################")
    for impute in (False, True):
        for c_idx in (False, True):
            for i_idx in (False, True):
                logger.info("#### TRAIN configs %d, instances %d" %
                            (int(c_idx), int(i_idx)))
                train_idx = indices[int(c_idx)][int(i_idx)]

                logger.info("#### Evaluating %s" % name)

                # Always reread data
                # Forget all data
                del enc_mat, inst_feat_dict, inst_ls, n_feats_used, perf_ls, \
                    to_ls, cen_ls

                logger.info("### Use cutoff %f and PAR%d" %
                            (args.cutoff, args.par))

                # Read in all data
                enc_mat, inst_feat_dict, inst_ls, n_feats_used, perf_ls, \
                    to_ls, cen_ls = load_data(
                        csv_file=args.target_algo_runs,
                        feature_file=args.feature_file,
                        pcs_file=args.pcs_file,
                        instance_file=args.instances,
                        normalize_features=False,
                        cutoff=args.cutoff,
                        oneHot=((name != "rfr") and (name != 'rfrq')),
                        par=args.par, downsample=DOWNSAMPLE,
                        quality=args.quality, normalize_parameters=False)

                # to_ls = numpy.array(to_ls)
                if cen_ls is not None:
                    cen_ls = numpy.array(cen_ls)

                # For sanity checks
                if data_hash is None:
                    # Not the best way to hash a numpy matrix
                    data_hash = hash(enc_mat.tostring())
                assert hash(enc_mat.tostring()) == data_hash
                inst_feat_dict_hash = hash(str(inst_feat_dict))

                feat_normalizer, trn_uncen_X, trn_uncen_y, trn_cen_X, trn_cen_y = \
                    new_preprocess_data(idx=train_idx,
                                        encoded_mat=enc_mat,
                                        inst_list=inst_ls,
                                        perf_list=perf_ls,
                                        cen_list=cen_ls,
                                        inst_feat_dict=inst_feat_dict,
                                        n_feats_used=n_feats_used,
                                        feature_normalizer=None)

                if impute:
                    # impute censored and timeouts
                    logger.info("Using logscale for train")
                    log10_train_uncen_y = warp(numpy.array(trn_uncen_y),
                                               quality=args.quality)
                    log10_train_cen_y = warp(numpy.array(trn_cen_y),
                                             quality=args.quality)
                    logger.info("### DO imputation")

                    # Impute data before training training folds
                    if trn_cen_X.shape[0] != 0 and not args.quality:
                        logger.info("Going to impute %d samples" %
                                    trn_cen_X.shape[0])

                        imputor = ImputorY(debug=False)

                        # NOTE
                        # par * cutoff defines the threshold, but as math works
                        # differently in logspace:
                        #   log(par*cutoff) = log(par) + log(cutoff)
                        # we need to calculate multiplicator, as imputator will
                        # always multiplicate

                        start = time.time()
                        log10_trn_imp_y = imputor.\
                            raw_impute_arrays(X_uncen=trn_uncen_X,
                                              y_uncen=log10_train_uncen_y,
                                              X_cen=trn_cen_X,
                                              y_cen=log10_train_cen_y,
                                              model=boot_model,
                                              par=warp(args.cutoff*args.par,
                                                       quality=args.quality) /
                                              warp(args.cutoff,
                                                   quality=args.quality),
                                              cutoff=warp(args.cutoff,
                                                          quality=args.quality),
                                              log=False, change_threshold=0.01,
                                              store_info=True)

                        for mode in ["train", "new_inst", "new_config", "both"]:
                            data_traj[mode].append(unwarp(imputor.data_traj,
                                                          quality=args.quality))
                            change_traj[mode].append(imputor.change_traj)
                        del imputor

                        dur = time.time() - start
                        logger.info("Imputation finished, took %fsec" % dur)

                        # Now stack the data together,
                        # call it >>>> trn_uncen_X, log10_train_uncen_y
                        trn_uncen_imp_X = numpy.vstack((trn_uncen_X, trn_cen_X))
                        log10_trn_uncen_imp_y = numpy.concatenate(
                                (log10_train_uncen_y, log10_trn_imp_y), axis=0)

                        # Build and Shuffle data
                        assert log10_trn_uncen_imp_y.shape[0] == \
                               trn_uncen_imp_X.shape[0]
                        shuffle_idx = rs.permutation(trn_uncen_imp_X.shape[0])

                        # Probably not the best way to name the imputed array
                        trn_uncen_X = trn_uncen_imp_X[shuffle_idx, :]
                        log10_train_uncen_y = log10_trn_uncen_imp_y[shuffle_idx]
                    else:
                        logger.critical("No data to impute")
                        for mode in ["train", "new_inst", "new_config", "both"]:
                            data_traj[mode].append(numpy.array([[1, 1], [1, 1]]))
                            change_traj[mode].append([1, 1])
                else:
                    logger.info("### Do NO imputation")
                    logger.info("Using logscale for train")
                    log10_train_uncen_y = warp(numpy.array(trn_uncen_y),
                                               quality=args.quality)

                # Train model
                # Make data PAR10 Data
                log10_train_uncen_y = numpy.array(log10_train_uncen_y)
                if not args.quality:
                    log10_c = warp(args.cutoff, quality=args.quality)
                    log10_thr = warp(args.cutoff*args.par, quality=args.quality)
                    logger.info("Mean/Max before PAR%ding %f/%f" %
                                (args.par, numpy.mean(log10_train_uncen_y),
                                 numpy.max(log10_train_uncen_y)))
                    log10_train_uncen_y[log10_train_uncen_y >= log10_c] = \
                        log10_thr

                    logger.info("Mean/Max after PAR%ding %f/%f" %
                                (args.par, numpy.mean(log10_train_uncen_y),
                                 numpy.max(log10_train_uncen_y)))
                else:
                    logger.info("Do not use PAR%d data" % args.par)

                dur, m = train_model(train_X=trn_uncen_X,
                                     train_y=log10_train_uncen_y,
                                     model=model)

                logger.info("Reread data to evaluate model")
                # Read in all data
                enc_mat, inst_feat_dict, inst_ls, n_feats_used, perf_ls, to_ls, cen_ls = \
                    load_data(csv_file=args.target_algo_runs,
                              feature_file=args.feature_file,
                              pcs_file=args.pcs_file,
                              instance_file=args.instances,
                              normalize_features=False,
                              cutoff=args.cutoff,
                              oneHot=((name != "rfr") and (name != 'rfrq')),
                              par=args.par, downsample=DOWNSAMPLE,
                              quality=args.quality, normalize_parameters=False)

                if cen_ls is not None:
                    to_ls = numpy.array(to_ls)
                    cen_ls = numpy.array(cen_ls)

                # For sanity checks
                assert hash(enc_mat.tostring()) == data_hash

                # Now predict on all Testdata
                for mode in ["train", "new_inst", "new_config", "both"]:
                    if mode == "train":
                        test_idx = indices[int(c_idx)][int(i_idx)]
                    elif mode == "new_inst":
                        test_idx = indices[int(c_idx)][int(not i_idx)]
                    elif mode == "new_config":
                        test_idx = indices[int(not c_idx)][int(i_idx)]
                    elif mode == "both":
                        test_idx = indices[int(not c_idx)][int(not i_idx)]
                    else:
                        raise ValueError("Don't know mode: %s" % mode)

                    logger.info("### TEST ON %s" % mode)

                    if mode not in test_uncen_ls:
                        # First loop
                        test_uncen_ls[mode] = list()
                        uncen_pred_ls[mode] = list()
                        uncen_trn_time[mode] = list()
                        imp_pred_ls[mode] = list()
                        imp_trn_time[mode] = list()
                        change_traj[mode] = list()
                        data_traj[mode] = list()

                    # Generate testdata
                    feat_normalizer, tst_uncen_X, tst_uncen_y, tst_cen_X, tst_cen_y = \
                        new_preprocess_data(idx=test_idx,
                                            encoded_mat=enc_mat,
                                            inst_list=inst_ls,
                                            perf_list=perf_ls,
                                            cen_list=cen_ls,
                                            inst_feat_dict=inst_feat_dict,
                                            n_feats_used=n_feats_used,
                                            feature_normalizer=feat_normalizer)
                    assert hash(str(inst_feat_dict)) == inst_feat_dict_hash

                    logger.info("Using logscale for test")
                    log10_tst_uncen_y = warp(numpy.array(tst_uncen_y),
                                             quality=args.quality)

                    if name == 'rfr':
                        log10_pred = m.predict(tst_uncen_X,
                                               logged=True)
                    elif name == "rfrq":
                        log10_pred = m.predict(tst_uncen_X, seed=0,
                                               num_samples=1)
                    else:
                        log10_pred = m.predict(tst_uncen_X)

                    rmse = wrap_rmse(y_true=log10_tst_uncen_y,
                                     y_pred=log10_pred)
                    logger.info("RMSE: %s" % rmse)

                    # Retransform performance data
                    logger.info("Remove logscale from prediction")
                    if impute:
                        imp_trn_time[mode].append(dur)
                        imp_pred_ls[mode].append(unwarp(log10_pred,
                                                        quality=args.quality))
                    else:
                         # store test data
                         # NOTE: Before we stored tst_uncen_X, now we store None
                         # to be backward compatible and save disc space
                        test_uncen_ls[mode].append([None, copy.deepcopy(tst_uncen_y)])
                        uncen_trn_time[mode].append(dur)
                        uncen_pred_ls[mode].append(unwarp(log10_pred,
                                                          quality=args.quality))

                    # Delete prediction
                    del log10_pred

    if args.save is not None:
        fh = open(args.save + "_all_modes_predictions.pkl", 'wb')
        pickle.dump(obj=(test_uncen_ls, uncen_pred_ls, uncen_trn_time,
                         imp_pred_ls, imp_trn_time, data_traj, change_traj),
                    file=fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.close()

if __name__ == "__main__":
    main()
