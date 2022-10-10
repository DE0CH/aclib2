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

from epm.reader import output_parser

from epm.preprocessing.pre_feature import PreprocessingFeatures

try:
    import cPickle as pickle
except ImportError:
    import pickle
import functools
import glob
import logging
import os
import sys
import time

import numpy

from epm.experiment_utils.data_handling import load_data, build_data, warp,\
    unwarp, separate_data_with_bools
from epm.preprocessing.fast_imputor_missing_y import FastImputorY as ImputorY
from epm.models import model_dict
from epm.pcs.config_space import ConfigSpace

logger = logging.getLogger("evaluate_regression_loro")
model_debug = False


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
    parser = argparse.\
        ArgumentParser(description='Train model on TRAIN data and save '
                                   'predictions on TEST data',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('target_algo_runs', metavar='JSON', nargs='+',
                        help='JSON files to read')
    parser.add_argument('--pcs', dest='pcs_file', default=None, required=True,
                        help='PCS file to read from')
    parser.add_argument('--features', dest='feature_file', required=False,
                        help='Feature file to read from')
    parser.add_argument('--instances', dest='instances', default=None,
                        help='Use instances given in that file')
    parser.add_argument('--random', dest="random", default=None,
                        help="Add random runs .json File")
    parser.add_argument('--model', dest='model', required=True,
                        choices=sorted(model_dict.model_dict.keys()),
                        help="Which model to use?")
    parser.add_argument('--cutoff', dest='cutoff', default=-1,
                        required=True, type=float,
                        help="What was the cutoff time?")
    parser.add_argument('--def', dest="default_folder", type=str,
                        default=None,
                        help="Path to SMAC validate-def folder")
    parser.add_argument('-v', dest='debug', default=False, action="store_true",
                        help="Show some more output")
    parser.add_argument('--save', dest='save', default=None,
                        help="Where to save predictions")
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help="Seed to use where applicable")
    parser.add_argument('--par', dest='par', default=1, type=int,
                        help="Treat TIMEOUTS as PARX")
    parser.add_argument('--quality', dest="quality", default=False,
                        action="store_true", help="Handle quality data?")
    parser.add_argument('--imputation', dest="imputation", default="def",
                        help="How to impute nonactive parameter")
    parser.add_argument('--downsample', dest="downsample", type=int,
                        default=300000,
                        help="Use at most that many configuration samples")
    parser.add_argument('--testpattern', dest="testpattern", type=str,
                        default=None, required=True,
                        help="Test on files matching this pattern an do not"
                             " train on files matching this pattern")
    parser.add_argument('--testdir', dest="testdir", type=str,
                        required=True, help="Test on all files in this "
                                            "directory that match testpattern")
    parser.add_argument('--dump', dest="dump", default=None,
                        required=False,
                        help="Dump imputed data and do not train a model")

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(levelname)s][%(module)s]'
                               '  %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

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

    if not os.path.isdir(os.path.dirname(args.testdir)):
        raise ValueError("%s is not a directory" % args.testdir)

    if args.instances is not None:
        if not os.path.isfile(args.instances):
            raise ValueError("%s is not a file" % args.instances)

    # Get Traindata and Testdata
    train_files = []
    test_files = []
    for f in args.target_algo_runs:
        if not args.testpattern in f:
            train_files.append(f)

    for f in os.listdir(args.testdir):
        if args.testpattern in f:
            test_files.append(os.path.join(args.testdir, f))

    logger.info("Train on files: %s" % str(train_files))
    logger.info("Test on files: %s" % str(test_files))

    ### Read in all data
    train_enc_mat, inst_feat_dict, train_inst_ls, n_feats_used, train_perf_ls,\
        train_to_ls, train_cen_ls = \
        load_data(train_files, feature_file=args.feature_file,
                  pcs_file=args.pcs_file, instance_file=args.instances,
                  normalize_features=False, cutoff=args.cutoff,
                  oneHot=False, par=args.par, downsample=args.downsample,
                  quality=args.quality, normalize_parameters=False)
    logger.info("ADD %d config runs" % len(train_perf_ls))
    ###

    ### Add random runs
    if args.random is not None:
        r_encoded_matrix, r_inst_feat_dict, r_inst_list, r_n_feats_used, \
            r_perf_list, r_timeout_list, r_cen_list = \
            load_data(csv_file=args.random, feature_file=args.feature_file,
                      pcs_file=args.pcs_file, instance_file=args.instances,
                      normalize_features=False, cutoff=args.cutoff,
                      oneHot=False, par=args.par, downsample=None,
                      impute_with=args.imputation, normalize_parameters=False,
                      quality=args.quality)
        logger.info("ADD %d random runs" % len(r_perf_list))
        train_enc_mat = numpy.vstack((train_enc_mat, r_encoded_matrix))
        # We don't need inst_feat_dict, it should be the same as we had before
        train_inst_ls.extend(r_inst_list)
        assert r_n_feats_used == n_feats_used
        train_perf_ls = numpy.hstack((train_perf_ls, r_perf_list))
        if not args.quality:
            train_to_ls.extend(r_timeout_list)
            train_cen_ls.extend(r_cen_list)
        del r_encoded_matrix, r_inst_feat_dict, r_inst_list, r_n_feats_used, \
            r_perf_list, r_timeout_list, r_cen_list
    ###

    ### Add default runs
    if args.default_folder is not None:
        val_obj_fl = os.path.join(args.default_folder,
                                  "validationObj*-cli-*-wal*rker.csv")
        try:
            val_obj_fl = glob.glob(val_obj_fl)[0]
        except IndexError:
            logger.error("Can't find %s" % val_obj_fl)
            raise
        val_str_fl = os.path.join(args.default_folder,
                                  "validationCall*-cli-*-wal*rker.csv")
        try:
            val_str_fl = glob.glob(val_str_fl)[0]
        except IndexError:
            logger.error("Can't find %s" % val_str_fl)
            raise

        def_conf, def_inst_list, def_perf_list = output_parser.\
            read_default_data(val_str_fl=val_str_fl, val_obj_fl=val_obj_fl,
                              pcs=args.pcs_file, impute_with=args.imputation,
                              normalize_parameters=False)
        logger.info("Add %d default performances" % len(def_inst_list))

        for idx, instance in enumerate(def_inst_list):
            train_enc_mat = numpy.vstack((train_enc_mat, def_conf))
            train_inst_ls.append(instance)
            train_perf_ls = numpy.hstack((train_perf_ls, def_perf_list[idx]))
            if def_perf_list[idx] >= args.cutoff:
                train_to_ls.append(True)
            else:
                train_to_ls.append(False)
            train_cen_ls.append(False)
    ###

    ### Feature preprocessing
    fpre = PreprocessingFeatures(inst_feats_dict=inst_feat_dict)
    inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)
    ###

    if not args.quality:
            logger.info("%d censored data found" % sum(train_cen_ls))
            logger.info("%d TIMEOUT data found" % sum(train_to_ls))

    ### Split into censored and uncensored
    if args.quality and train_cen_ls is None:
        train_uncen_enc_mat = train_enc_mat
        train_uncen_inst_ls = train_inst_ls
        train_uncen_perf_ls = train_perf_ls
    else:
        train_cen_enc_mat, train_cen_inst_ls, train_cen_perf_ls, \
               train_uncen_enc_mat, train_uncen_inst_ls, train_uncen_perf_ls = \
            separate_data_with_bools(data_matrix=train_enc_mat,
                                     inst_list=train_inst_ls,
                                     perf_list=train_perf_ls,
                                     succ_list=train_cen_ls)

    del train_enc_mat, train_inst_ls, train_perf_ls, train_cen_ls, train_to_ls

    ### Build Matrix
    logger.info("Using logscale for train")
    train_cen_X = list()
    if not args.quality:
        train_cen_y = warp(train_cen_perf_ls, quality=args.quality)
        train_cen_X = build_data(data_matrix=train_cen_enc_mat,
                                 inst_list=train_cen_inst_ls,
                                 inst_feat_dict=inst_feat_dict,
                                 n_feats=n_feats_used)

    train_uncen_X = build_data(data_matrix=train_uncen_enc_mat,
                               inst_list=train_uncen_inst_ls,
                               inst_feat_dict=inst_feat_dict,
                               n_feats=n_feats_used)

    train_uncen_y = warp(train_uncen_perf_ls, quality=args.quality)
    ###

    ### Prepare Model
    name, model, boot_model = model_dict.model_dict[args.model]

    if name == "rfrq":
        cs = ConfigSpace(pcs_file=args.pcs_file, normalize=False)
        boot_model = functools.partial(boot_model.func, cs=cs,
                                       n_feats=n_feats_used,
                                       **boot_model.keywords)
        model = functools.partial(model.func, cs=cs,
                                  n_feats=n_feats_used,
                                  **model.keywords)
    else:
        raise ValueError("Don't know that model")
    ###

    logger.info("###############################################")
    logger.info("### TRAIN on %s data" % str(train_uncen_X.shape))
    logger.info("### IMPUTE %s data" % str(len(train_cen_X)))
    logger.info("### Evaluating %s" % name)
    logger.info("### Use cutoff %f and PAR%d" % (args.cutoff, args.par))

    ### Imputation
    if len(train_cen_X) > 0 and not args.quality:
        logger.info("### DO imputation")

        # Impute data before training training folds
        imputor = ImputorY(debug=False)

        # NOTE
        # par * cutoff defines the threshold, but as math works
        # differently in logspace:
        #   log(par*cutoff) = log(par) + log(cutoff)
        # we need to calculate multiplicator, as imputator will
        # always multiplicate

        start = time.time()
        train_imp_y = \
            imputor.raw_impute_arrays(
                    X_uncen=train_uncen_X,
                    y_uncen=train_uncen_y,
                    X_cen=train_cen_X,
                    y_cen=train_cen_y,
                    model=boot_model,
                    par=warp(args.cutoff*args.par, quality=args.quality) /
                        warp(args.cutoff,  quality=args.quality),
                    cutoff=warp(args.cutoff, quality=args.quality),
                    log=False, change_threshold=0.01,
                    store_info=True)
        del imputor
        dur = time.time() - start
        logger.info("Imputation finished, took %fsec" % dur)

        # Now stack the data together,
        # call it >>>> trn_uncen_X, log10_train_uncen_y
        train_imp_X = numpy.vstack((train_uncen_X, train_cen_X))
        train_imp_y = numpy.concatenate((train_uncen_y, train_imp_y), axis=0)

        # Build and Shuffle data
        shuffle_idx = rs.permutation(train_imp_X.shape[0])

        # Probably not the best way to name the imputed array
        train_imp_X = train_imp_X[shuffle_idx, :]
        train_imp_y = train_imp_y[shuffle_idx]
    else:
        logger.info("###NO imputation")
        logger.info("Using logscale for train")
        train_imp_y = warp(numpy.array(train_uncen_y), quality=args.quality)
        train_imp_X = train_uncen_X
    ###

    ### Make data PAR10 Data
    train_imp_y = numpy.array(train_imp_y)
    if not args.quality:
        log10_c = warp(args.cutoff, quality=args.quality)
        log10_thr = warp(args.cutoff*args.par, quality=args.quality)
        logger.info("Mean/Max before PAR%ding %f/%f" %
                    (args.par, numpy.mean(train_imp_y),
                     numpy.max(train_imp_y)))
        train_imp_y[train_imp_y >= log10_c] = log10_thr
        logger.info("Mean/Max after PAR%ding %f/%f" %
                    (args.par, numpy.mean(train_imp_y),
                     numpy.max(train_imp_y)))
    else:
        logger.info("Do not use PAR%d data" % args.par)
    ###

    ### Train model
    dur, m = train_model(train_X=train_imp_X,
                         train_y=train_imp_y,
                         model=model)
    ###

    ### Prepare test data
    test_enc_mat, _, test_inst_ls, _, test_perf_ls, _, test_cen_ls = \
        load_data(csv_file=test_files,
                  feature_file=args.feature_file,
                  pcs_file=args.pcs_file,
                  instance_file=args.instances,
                  normalize_features=False,
                  cutoff=args.cutoff,
                  oneHot=False, par=args.par, downsample=None,
                  quality=args.quality, normalize_parameters=False)

    if args.quality and test_cen_ls is None:
        test_uncen_enc_mat = test_enc_mat
        test_uncen_inst_ls = test_inst_ls
        test_uncen_perf_ls = test_perf_ls
    else:
        _, _, _, test_uncen_enc_mat, test_uncen_inst_ls, test_uncen_perf_ls = \
            separate_data_with_bools(data_matrix=test_enc_mat,
                                     inst_list=test_inst_ls,
                                     perf_list=test_perf_ls,
                                     succ_list=test_cen_ls)

    del test_enc_mat, test_inst_ls, test_perf_ls, test_cen_ls

    test_X = build_data(data_matrix=test_uncen_enc_mat,
                        inst_list=test_uncen_inst_ls,
                        inst_feat_dict=inst_feat_dict,
                        n_feats=n_feats_used)
    logger.info("Using logscale for train")
    test_y = numpy.array(test_uncen_perf_ls)
    ###

    ### Do predictions
    pred_y = numpy.array(m.predict(test_X, seed=0, num_samples=1))

    if not args.quality:
        pred_y[pred_y >= warp(args.cutoff, quality=args.quality)] = \
            warp(args.cutoff*args.par, quality=args.quality)
        test_y[test_y >= args.cutoff] = args.cutoff*args.par
    pred_y = unwarp(pred_y, quality=args.quality)
    ###

    if args.dump:
        with open(args.dump, 'w') as fh:
            pickle.dump((train_imp_X, train_imp_y, test_X, test_y), fh)
        print("Dumped data")
        sys.exit(1)

    if args.save is not None:
        fh = open(args.save + "_predictions.pkl", 'wb')
        pickle.dump(obj=(test_y, pred_y), file=fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
        fh.close()

if __name__ == "__main__":
    main()
