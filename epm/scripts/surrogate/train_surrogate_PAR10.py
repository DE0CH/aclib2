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

import os
import functools
import glob
import logging
import numpy as np
import time
import json_tricks
from pathlib import Path
from ConfigSpace.read_and_write import pcs
from sklearn.decomposition.pca import PCA
from epm.experiment_utils.data_handling import load_data, \
    separate_data_with_bools, warp
from epm.models.model_dict import model_dict as model_dict
from epm.reader import output_parser
from epm.experiment_utils import data_handling
from epm.preprocessing.rfr_imputator import RFRImputator
from epm.models.external_rfr import save_model

logger = logging.getLogger("trainSurrogate_PAR10")
logger.setLevel(logging.INFO)

model_debug = False
imputor_debug = False


def pca_inst_feat(inst_feat_dict, num_components=7):
    logger.critical("Do a PCA and return %d features" % num_components)

    key_list = list()
    features = list()
    for k in inst_feat_dict.keys():
        key_list.append(k)
        features.append(inst_feat_dict[k])

    features = np.array(features)
    pca_model = PCA(n_components=num_components)

    features = pca_model.fit_transform(features)

    n_inst_feat_dict = dict()
    for idx, k in enumerate(key_list):
        assert len(features[idx, :]) == num_components
        n_inst_feat_dict[k] = features[idx, :]

    return n_inst_feat_dict, num_components


def parse_args():
    parser = argparse. \
        ArgumentParser(description='Impute and train a model on data',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pcs', dest='pcs_file', default=None, required=True,
                        help='PCS file to read from')
    parser.add_argument('--features', dest='feature_file', required=False,
                        help='Feature file to read from')
    parser.add_argument('--instances', dest='instances', default=None,
                        help='Use instances given in that file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=sorted(model_dict.keys()),
                        help="Which model to use?")
    parser.add_argument('-v', dest='debug', default=False, action="store_true",
                        help="Show some more output")
    parser.add_argument('--save', dest='save', default=None, required=True,
                        help="Path to save model")
    parser.add_argument('--name', dest='name', default='out', required=False,
                        type=str, help='Prefix of file in which epm is stored.')
    parser.add_argument('--cutoff', dest='cutoff', required=True, type=float,
                        help="What was the cutoff time?")
    parser.add_argument('target_algo_runs', metavar='JSON', nargs='+',
                        help='JSON file(s) to use for training')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help="Seed to used for randomState")
    parser.add_argument('--par', dest='par', default=1, type=int,
                        help="Treat TIMEOUTS as PARX")
    parser.add_argument('--def', dest="default_folder", type=str,
                        default=None, help="Path to SMAC validate-def folder")
    parser.add_argument('--noimp', dest="noimputation", default=False,
                        action="store_true",
                        help="Don't impute and train only on uncensored+timeout")
    parser.add_argument('--random', dest="random", default=None,
                        help="Add random runs .json File")
    parser.add_argument('--imputation', dest="imputation", default="def",
                        help="How to impute nonactive parameter")
    parser.add_argument('--pca', dest="pca", default=False, action="store_true",
                        help="Perform a PCA on instance features using 7"
                             " components")
    parser.add_argument('--downsample', dest="downsample", type=int,
                        default=200000,
                        help="Use at most that many configuration samples. "
                             "To use all available data points, set to -1")
    parser.add_argument('--quality', dest="quality", default=False,
                        action="store_true", help="Handle quality data?")
    args, unknown = parser.parse_known_args()

    return args, unknown


def main():
    args, unknown = parse_args()

    # Take all values and don't downsample
    if args.downsample == -1:
        args.downsample = None

    if args.debug:
        logger.setLevel(logging.DEBUG)

    rs = np.random.RandomState(args.seed)
    name, model, boot_model = model_dict[args.model]

    assert Path(args.pcs_file).is_file()
    assert Path(args.instances).is_file() \
        if args.instances is not None else True
    if not Path(args.save).exists():
        Path(args.save).mkdir(exist_ok=True, parents=True)

    # Read in all data
    encoded_matrix, inst_feat_dict, inst_list, \
    n_feats_used, perf_list, timeout_list, cen_list = \
        load_data(csv_file=args.target_algo_runs,
                  feature_file=args.feature_file,
                  pcs_file=args.pcs_file, instance_file=args.instances,
                  normalize_features=True, cutoff=args.cutoff,
                  oneHot=((name != "rfr") and (name != 'rfrq')), par=args.par,
                  downsample=args.downsample, impute_with=args.imputation,
                  normalize_parameters=False, quality=args.quality)

    logger.info('Read in ConfigSpaace from {}'.format(args.pcs_file))
    cs = pcs.read(open(args.pcs_file, 'r'))

    if not args.quality:
        logger.info("%d censored data found" % sum(cen_list))
        logger.info("%d TIMEOUT data found" % sum(timeout_list))
    logger.info("Read %d data" % len(perf_list))

    # ADD default runs
    if args.default_folder is not None:
        val_obj_fl = os.path. \
            join(args.default_folder,
                 "validationObjectiveMatrix-cli-*-walltimeworker.csv")
        try:
            val_obj_fl = glob.glob(val_obj_fl)[0]
        except IndexError as e:
            logger.error("Can't find %s" % val_obj_fl)
            raise e

        val_str_fl = os.path. \
            join(args.default_folder,
                 "validationCallStrings-cli-*-walltimeworker.csv")
        try:
            val_str_fl = glob.glob(val_str_fl)[0]
        except IndexError as e:
            logger.error("Can't find %s" % val_str_fl)
            raise e

        def_conf, def_inst_list, def_perf_list = output_parser. \
            read_default_data(val_str_fl=val_str_fl, val_obj_fl=val_obj_fl,
                              pcs=args.pcs_file, impute_with=args.imputation,
                              normalize_parameters=False)
        logger.info("ADD %d performances for default config on instances" %
                    len(def_inst_list))

        for idx, instance in enumerate(def_inst_list):
            encoded_matrix = np.vstack((encoded_matrix, def_conf))
            inst_list.append(instance)
            perf_list = np.hstack((perf_list, def_perf_list[idx]))
            if not args.quality:
                if def_perf_list[idx] >= args.cutoff:
                    timeout_list.append(True)
                    cen_list.append(False)
                else:
                    timeout_list.append(False)
                    cen_list.append(False)
            else:
                assert timeout_list is None
                assert cen_list is None

    # ADD random runs
    if args.random is not None:
        r_encoded_matrix, r_inst_feat_dict, r_inst_list, r_n_feats_used,\
            r_perf_list, r_timeout_list, r_cen_list = \
            load_data(csv_file=args.random, feature_file=args.feature_file,
                      pcs_file=args.pcs_file, instance_file=args.instances,
                      normalize_features=True, cutoff=args.cutoff,
                      oneHot=((name != "rfr") and (name != 'rfrq')),
                      par=args.par, quality=args.quality,
                      downsample=None, impute_with=args.imputation,
                      normalize_parameters=False)
        logger.info("ADD %d random runs" % len(r_perf_list))
        encoded_matrix = np.vstack((encoded_matrix, r_encoded_matrix))
        # We don't need inst_feat_dict, it should be the same as we had before
        inst_list.extend(r_inst_list)
        assert r_n_feats_used == n_feats_used
        perf_list = np.hstack((perf_list, r_perf_list))
        if not args.quality:
            timeout_list.extend(r_timeout_list)
            cen_list.extend(r_cen_list)
        del r_encoded_matrix, r_inst_feat_dict, r_inst_list, r_n_feats_used, \
            r_perf_list, r_timeout_list, r_cen_list

    if args.pca:
        logger.critical("Perform a PCA, keep 7 components")
        inst_feat_dict, n_feats_used = pca_inst_feat(inst_feat_dict,
                                                     num_components=7)

    logger.info("Having a total of %d runs" % len(perf_list))
    logger.info("Feature dimension: %d" % len(list(inst_feat_dict.values())[0]))

    if not args.quality:
        timeout_list = np.array(timeout_list)
        cen_list = np.array(cen_list)

    if name == "rfr":
        boot_model = functools.partial(boot_model.func, cs=cs,
                                       n_feats=n_feats_used,
                                       **boot_model.keywords)
        model = functools.partial(model.func, cs=cs, n_feats=n_feats_used,
                                  cutoff=warp(args.cutoff,
                                              quality=args.quality),
                                  threshold=warp(args.cutoff * 2,
                                                 quality=args.quality),
                                  **model.keywords)
    elif name == "rfrq":
        boot_model = functools.partial(boot_model.func, cs=cs,
                                       n_feats=n_feats_used,
                                       **boot_model.keywords)
        model = functools.partial(model.func, cs=cs, n_feats=n_feats_used,
                                  **model.keywords)

    logger.debug("Model arguments %s" % str(model.keywords))

    # From here everything diverges for impute and not impute
    if not args.noimputation and not args.quality:
        train_cen_encoded_matrix, train_cen_inst_list, train_cen_y, \
         train_uncen_encoded_matrix, train_uncen_inst_list, \
         train_uncen_y = \
         separate_data_with_bools(data_matrix=encoded_matrix,
                                  inst_list=inst_list,
                                  perf_list=perf_list,
                                  succ_list=cen_list)

        logger.info("###############################################")
        logger.info("## BUILD DATA %s" % name)

        # Build datasets
        train_uncen_X = data_handling. \
            build_data(data_matrix=train_uncen_encoded_matrix,
                       inst_list=train_uncen_inst_list,
                       inst_feat_dict=inst_feat_dict,
                       n_feats=n_feats_used)
        train_cen_X = data_handling. \
            build_data(data_matrix=train_cen_encoded_matrix,
                       inst_list=train_cen_inst_list,
                       inst_feat_dict=inst_feat_dict,
                       n_feats=n_feats_used)

        logger.info("Uncen data dimension: %s" % str(train_uncen_X.shape))
        logger.info("Cen data dimension: %s" % str(train_cen_X.shape))

        # Put performance data on logscale
        logger.critical("Using logscale for train")
        log10_train_uncen_y = warp(np.array(train_uncen_y),
                                   quality=args.quality)
        log10_train_cen_y = warp(np.array(train_cen_y),
                                 quality=args.quality)

        del train_uncen_y, train_cen_y

        logger.info("min/max uncen before imputation: %f/%f" %
                    (np.min(log10_train_uncen_y, axis=0),
                     np.max(log10_train_uncen_y, axis=0)))
        if train_cen_X.shape[0] > 0:
            logger.info("### IMPUTATION")
            logger.info("Going to impute %d samples, %s" %
                        (train_cen_X.shape[0], str(train_cen_X.shape)))

            logger.info("min/max cen before imputation: %f/%f" %
                        (np.min(log10_train_cen_y, axis=0),
                         np.max(log10_train_cen_y, axis=0)))

            # Impute data for training folds
            imp_start = time.time()

            imputor = RFRImputator(
                    rng=np.random.RandomState(),
                    cutoff=warp(args.cutoff, quality=args.quality),
                    threshold=warp(args.cutoff * args.par,
                                   quality=args.quality),
                    model=boot_model(),
                    change_threshold=0.1,
                    max_iter=5
                    )

            log10_train_imp_y = imputor.impute(
                    censored_X=train_cen_X,
                    censored_y=log10_train_cen_y,
                    uncensored_X=train_uncen_X,
                    uncensored_y=log10_train_uncen_y
                    )
            log10_train_imp_y = log10_train_imp_y.reshape((-1,))
            log10_train_uncen_y = log10_train_uncen_y.reshape((-1,))

            del imputor
            imp_dur = time.time() - imp_start

            logger.info("min/max uncen after imputation: %f/%f" %
                        (np.min(log10_train_uncen_y, axis=0),
                         np.max(log10_train_uncen_y, axis=0)))
            logger.info("min/max imp after imputation: %f/%f" %
                        (np.min(log10_train_imp_y, axis=0),
                         np.max(log10_train_imp_y, axis=0)))

            logger.info("Imputation took %g sec" % imp_dur)

            # Now stack all (uncen, imp) data together
            train_uncen_imp_X = np.vstack((train_uncen_X, train_cen_X))
            log10_train_uncen_imp_y = np.concatenate((log10_train_uncen_y,
                                                      log10_train_imp_y),
                                                     axis=0)
        else:
            raise ValueError("### NO IMPUTATION necessary")

    elif args.noimputation or args.quality:
        # raise NotImplementedError()
        logger.info("Do NOT impute")
        if not args.quality:
            train_cen_encoded_matrix, train_cen_inst_list, train_cen_y, \
            train_uncen_encoded_matrix, train_uncen_inst_list, train_uncen_y = \
                separate_data_with_bools(data_matrix=encoded_matrix,
                                         inst_list=inst_list,
                                         perf_list=perf_list,
                                         succ_list=cen_list)
            del train_cen_encoded_matrix, train_cen_inst_list, train_cen_y
        else:
            # If we have quality data we don't have censored data+
            train_uncen_encoded_matrix = encoded_matrix
            train_uncen_inst_list = inst_list
            train_uncen_y = perf_list

        logger.info("###############################################")
        logger.info("## BUILD DATA %s" % name)

        # Build datasets
        train_uncen_imp_X = data_handling. \
            build_data(data_matrix=train_uncen_encoded_matrix,
                       inst_list=train_uncen_inst_list,
                       inst_feat_dict=inst_feat_dict,
                       n_feats=n_feats_used)
        # Put performance data on logscale
        logger.critical("Using logscale for train")
        log10_train_uncen_imp_y = warp(np.array(train_uncen_y),
                                       quality=args.quality)

    logger.info("### TRAINING")
    logger.info("Start training (%s)" %
                str(train_uncen_imp_X.shape))

    # Build and Shuffle data
    assert log10_train_uncen_imp_y.shape[0] == train_uncen_imp_X.shape[0]
    shuffle_idx = rs.permutation(train_uncen_imp_X.shape[0])
    train_uncen_imp_X = train_uncen_imp_X[shuffle_idx, :]
    log10_train_uncen_imp_y = log10_train_uncen_imp_y[shuffle_idx]

    # Make PAR 10 Data
    logger.critical("PUT DATA ON PAR%d scale" % args.par)
    logger.info("min/max/mean before par10: {}/{}/{}".format(
            np.min(log10_train_uncen_imp_y, axis=0),
            np.max(log10_train_uncen_imp_y, axis=0),
            np.mean(log10_train_uncen_imp_y, axis=0)))

    log10_train_uncen_imp_y[log10_train_uncen_imp_y >=
                            warp(args.cutoff, quality=args.quality)] = \
        warp(args.cutoff * args.par, quality=args.quality)

    logger.info("min/max/mean after par10: {}/{}/{}".format(
            np.min(log10_train_uncen_imp_y, axis=0),
            np.max(log10_train_uncen_imp_y, axis=0),
            np.mean(log10_train_uncen_imp_y, axis=0)))

    logger.info(
        "Uncen+Imp data dimension: %s" % str(train_uncen_imp_X.shape))


    # Fit a model
    start = time.time()
    m = model()
    if args.imputation == "outlier":
        # We have to change type array here
        m.change_type_array_to_outlier_imputation()

    m.fit(train_uncen_imp_X, log10_train_uncen_imp_y)
    dur = time.time() - start
    logger.info("Training model took %s" % dur)

    if args.save is not None:

        if args.pca:
            save_fn = ".PCA7"
        else:
            save_fn = '.' + args.name
        if args.imputation == "outlier":
            save_fn += ".par%d.%s" % (args.par, args.imputation)
        else:
            save_fn += ".par%d" % args.par
        if args.default_folder is not None:
            save_fn += ".def"
        if args.noimputation:
            save_fn += ".noImp"
        if args.random is not None:
            save_fn += ".random"

        # save data to file.
        args.save = Path(args.save)

        json_file = args.save / ('inst_feat_dict' + save_fn + '.json')
        json_tricks.dump(fp=json_file.open('w'),
                         obj=inst_feat_dict,
                         indent=4)

        # Store the config space also to cs to avoid type problems.
        cs_file = args.save / ('config_space' + save_fn + '.pcs')
        cs_file.write_text(pcs.write(cs))

        save_model(wrapper=m,
                   save_path=args.save,
                   model_name='pyrfr_model' + save_fn + '.bin',
                   wrapper_name='pyrfr_wrapper' + save_fn + '.pkl')

        logger.info("Saved: model%s to %s" % (save_fn, str(args.save)))

    logger.info('FINISHED RUN')


if __name__ == "__main__":
    main()
