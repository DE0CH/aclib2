#!/usr/local/bin/python2.7
# encoding: utf-8
"""
EPM -- emperical performance models

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
import collections
import copy
import logging
import os

from tabulate import tabulate
import numpy

import epm.experiment_utils.script_helper as script_helper


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on data')
    parser.add_argument('--pkl', dest='pkl', default=None,
                        help='pkl file to read from')
    parser.add_argument('--cutoff', dest="cutoff", default=100000, type=float,
                        help="Replace values higher than this")
    parser.add_argument('--par', dest="par", default=1, type=int,
                        help="Replace values >cutoff with PAR*cutoff")
    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.DEBUG)
    if not os.path.isfile(args.pkl):
        raise ValueError("Does not exist: %s" % args.pkl)
    else:
        logging.info("Loading %s" % args.pkl)

    crossval_uncen_data_test, uncen_pred_list, uncen_train_time, imp_pred_list, imp_train_time, data_traj, change_traj = pickle.load(open(args.pkl, 'rb'))

    metrics = (("RMSE", script_helper.wrap_rmse),
               ("CC", script_helper.wrap_spearman))

    uncen_metric_dict = collections.OrderedDict()
    imp_metric_dict = collections.OrderedDict()

    for mode in ("train", "new_inst", "new_config", "both"):
        uncen_metric_dict[mode] = dict()
        imp_metric_dict[mode] = dict()

        for m in metrics:
            uncen_metric_dict[mode][m[0]] = list()
            imp_metric_dict[mode][m[0]] = list()

        for f in range(len(crossval_uncen_data_test[mode])):
            true_values = numpy.array(copy.deepcopy(crossval_uncen_data_test[mode][f][1]))
            pred_values = copy.deepcopy(uncen_pred_list[mode][f])

            # Replace values higher than cutoff
            true_values[true_values >= args.cutoff] = args.cutoff*args.par
            pred_values[pred_values >= args.cutoff] = args.cutoff*args.par

            # Calc metric
            for m_name, m_func in metrics:
                uncen_metric_dict[mode][m_name].append(
                        m_func(y_true=numpy.log10(true_values),
                               y_pred=numpy.log10(pred_values)))

        for f in range(len(crossval_uncen_data_test[mode])):

            true_values = numpy.array(copy.deepcopy(crossval_uncen_data_test[mode][f][1]))
            pred_values = copy.deepcopy(imp_pred_list[mode][f])

            # Replace values higher than cutoff
            true_values[true_values >= args.cutoff] = args.cutoff*args.par
            pred_values[pred_values >= args.cutoff] = args.cutoff*args.par

            for m_name, m_func in metrics:
                imp_metric_dict[mode][m_name].append(
                        m_func(y_true=numpy.log10(true_values),
                               y_pred=numpy.log10(pred_values)))

    tab_dict = collections.OrderedDict()
    tab_dict[" "] = ["not imputed", "imputed"]
    num_fmt = "$%4.2f \mp %4.2f$"
    n_round = 2
    for mode in ("train", "new_inst", "new_config", "both"):
        imp_mean_std = num_fmt % \
                       (round(numpy.mean(imp_metric_dict[mode]["RMSE"]), n_round),
                        round(numpy.std(imp_metric_dict[mode]["RMSE"]), n_round))
        uncen_std = num_fmt % \
                    (round(numpy.mean(uncen_metric_dict[mode]["RMSE"]), n_round),
                     round(numpy.std(uncen_metric_dict[mode]["RMSE"]), n_round))

        tab_dict["%s_RMSE" % mode] = [uncen_std, imp_mean_std]

        imp_mean_std = num_fmt % \
                       (round(numpy.mean(imp_metric_dict[mode]["CC"]), n_round),
                        round(numpy.std(imp_metric_dict[mode]["CC"]), n_round))
        uncen_std = num_fmt % \
                    (round(numpy.mean(uncen_metric_dict[mode]["CC"]), n_round),
                     round(numpy.std(uncen_metric_dict[mode]["CC"]), n_round))

        tab_dict["%s_CC" % mode] = [uncen_std, imp_mean_std]

    a = tabulate(tab_dict, headers="keys", tablefmt="latex")
    a = a.replace('\\$', '$')
    a = a.replace(r'\textbackslash{}', '\\')
    print(a)

if __name__ == "__main__":
    main()