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
import copy
import glob
import logging
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout, subplots_adjust, subplot
import numpy
from matplotlib import rc
from tabulate import tabulate

from epm.experiment_utils.data_handling import load_data, partition_data
from epm.experiment_utils.script_helper import wrap_rmse, wrap_spearman
from epm.reader import output_parser

logger = logging.getLogger("scatter_regression_meani_meanc")


def get_handle_lists(l):
    """Copied from here:
    http://nbviewer.jupyter.org/gist/leejjoon/5603703
    returns a list of lists of handles.
    """
    tree = l._legend_box.get_children()[1]

    for column in tree.get_children():
        for row in column.get_children():
            yield row.get_children()[0].get_children()


def main():
    DOWNSAMPLE = 200000
    rc('text', usetex=True)
    dpi = 600

    parser = argparse.ArgumentParser(description='Evaluate a model on data')
    parser.add_argument('--pkl', dest='pkl', default=None,
                        help='pkl file to read from')
    parser.add_argument('--save', dest='save', default=None,
                        help="Where to save plot")
    parser.add_argument('--cutoff', dest="cutoff", default=100000, type=float,
                        help="Replace values higher than this")
    parser.add_argument('--par', dest="par", default=1, type=int,
                        help="Replace values >cutoff with PAR*cutoff")
    parser.add_argument('target_algo_runs', metavar='JSON', nargs='+',
                        help='JSON files to read')
    parser.add_argument('--pcs', dest='pcs_file', default=None, required=True,
                        help='PCS file to read from')
    parser.add_argument('--features', dest='feature_file', required=True,
                        help='Feature file to read from')
    parser.add_argument('--instances', dest='instances', default=None,
                        help='Use instances given in that file')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help="USE SAME SEED AS evaluate_regression_ni_nc.py")
    parser.add_argument('--def', dest="default_folder", required=True, type=str,
                        help="Path to SMAC validate-def folder")
    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(args.default_folder):
        raise ValueError("DOES NOT EXIST: %s" % args.default_folder)
    rs = numpy.random.RandomState(args.seed)

    # We first read smac-validate output
    val_obj_fl = os.path.join(
            args.default_folder,
            "validationObjectiveMatrix-cli-*-walltimeworker.csv")
    val_obj_fl = glob.glob(val_obj_fl)[0]
    perf_def = output_parser.\
        read_validationObjectiveMatrix_file(val_obj_fl)

    ### We have to reread data in order to know instances/configs
    # Read in all data. No need to do oneHot, because data will be reread anyway
    enc_mat, inst_feat_dict, inst_ls, n_feats_used, perf_ls, to_ls, cen_ls = \
        load_data(args.target_algo_runs, feature_file=args.feature_file,
                  pcs_file=args.pcs_file, instance_file=args.instances,
                  normalize_features=False, cutoff=args.cutoff,
                  oneHot=False, par=args.par, downsample=DOWNSAMPLE)

    logger.info("%d censored data found" % sum(cen_ls))

    # Split data in four parts
    indices = partition_data(data_matrix=enc_mat, inst_list=inst_ls, rs=rs)

    # Remove censored points
    for c_idx in (False, True):
        for i_idx in (False, True):
            tmp = list()
            for i in indices[c_idx][i_idx]:
                if not cen_ls[i]:
                    tmp.append(i)
            indices[c_idx][i_idx] = tmp

    if not os.path.isfile(args.pkl):
        raise ValueError("Does not exist: %s" % args.pkl)
    else:
        logging.info("Loading %s" % args.pkl)

    crossval_uncen_data_test, uncen_pred_list, _, imp_pred_list, _, _, _ = \
        pickle.load(open(args.pkl, 'r'))

    for grid_idx, mode in enumerate(["both", ]):  # ["train", "both"]):
        logger.critical("Processing %s" % mode)

        if mode == "train":
            bool_array = ((0, 0), (0, 1), (1, 0), (1, 1))
        elif mode == "both":
            bool_array = ((1, 1), (1, 0), (0, 1), (0, 0))

        ax = subplot(aspect='equal')
        ax.grid(True, linestyle='-', linewidth=1, which='major',
                color='gray', alpha=0.5, zorder=99)

        rmse_worse = list()
        rmse_better = list()
        rmse_all = list()

        cc_worse = list()
        cc_better = list()
        cc_all = list()

        for ls_idx, b_idx in enumerate(bool_array):
            # Copying might not be necessary, but we do it nevertheless
            ind = copy.deepcopy(indices[b_idx[0]][b_idx[1]])
            pred = numpy.array(copy.deepcopy(imp_pred_list[mode][ls_idx]))
            real = numpy.array(copy.deepcopy(
                    crossval_uncen_data_test[mode][ls_idx][1]))
            current_inst_ls = [inst_ls[i] for i in ind]

            # Some checks
            assert len(imp_pred_list[mode][ls_idx]) == len(indices[b_idx[0]][b_idx[1]]),\
                "%d != %d" % (len(imp_pred_list[mode][ls_idx]),
                              len(indices[b_idx[0]][b_idx[1]]))
            assert len(pred) == len(ind), "%d != %d" % (len(pred), len(ind))
            assert len(pred) == len(real), "%d != %d" % (len(pred), len(real))
            assert len(current_inst_ls) == len(pred), \
                "%d != %d" % (len(current_inst_ls), len(pred))

            pred[pred >= args.cutoff] = args.cutoff*args.par
            real[real >= args.cutoff] = args.cutoff*args.par

            # Find performance of best configuration for each instance
            perf_inc = dict()
            for idx, instance in enumerate(current_inst_ls):
                if instance not in perf_inc:
                    perf_inc[instance] = perf_ls[idx]
                else:
                    perf_inc[instance] = min(perf_inc[instance], perf_ls[idx])

            worse_idx = list()
            better_idx = list()
            same_idx = list()

            lambd = 0.1
            for idx, instance in enumerate(current_inst_ls):
                gap = max(0, perf_def[instance] - perf_inc[instance])
                if real[idx] <= (perf_inc[instance] + (gap * lambd)):
                    better_idx.append(True)
                    worse_idx.append(False)
                    same_idx.append(False)
                elif real[idx] > (perf_inc[instance] + gap):
                    better_idx.append(False)
                    worse_idx.append(True)
                    same_idx.append(False)
                else:
                    better_idx.append(False)
                    worse_idx.append(False)
                    same_idx.append(True)

            better_idx = numpy.array(better_idx, dtype=bool)
            worse_idx = numpy.array(worse_idx, dtype=bool)
            same_idx = numpy.array(same_idx, dtype=bool)

            if numpy.sum(worse_idx) > 0:
                rmse_worse.append(wrap_rmse(
                        y_true=numpy.log10(real[worse_idx]),
                        y_pred=numpy.log10(pred[worse_idx])))
                cc_worse.append(wrap_spearman(
                        y_true=numpy.log10(real[worse_idx]),
                        y_pred=numpy.log10(pred[worse_idx])))
            else:
                logging.critical("No worse values found")
            if numpy.sum(better_idx) > 0:
                rmse_better.append(wrap_rmse(
                        y_true=numpy.log10(real[better_idx]),
                        y_pred=numpy.log10(pred[better_idx])))
                cc_better.append(wrap_spearman(
                        y_true=numpy.log10(real[better_idx]),
                        y_pred=numpy.log10(pred[better_idx])))
            else:
                logging.critical("No better values found")

            rmse_all.append(wrap_rmse(
                    y_true=numpy.log10(real),
                    y_pred=numpy.log10(pred)))
            cc_all.append(wrap_spearman(
                        y_true=numpy.log10(real),
                        y_pred=numpy.log10(pred)))

            if ls_idx == len(bool_array) - 1:
                # Subsample to 5000 datapoints
                plot_idx = list(range(0, pred.shape[0]))
                rs.shuffle(plot_idx)
                plot_idx = plot_idx[:5000]

                better_idx = better_idx[plot_idx]
                worse_idx = worse_idx[plot_idx]
                same_idx = same_idx[plot_idx]

                real = real[plot_idx]
                pred = pred[plot_idx]

                # Scatter actual points
                ax.scatter(x=real[better_idx],
                           y=pred[better_idx],
                           facecolor="#2c7bb6", c='', edgecolor="",
                           alpha=0.7, s=4, label="high performing")
                ax.scatter(x=real[worse_idx],
                           y=pred[worse_idx],
                           facecolor="#d7191c", c='', edgecolor="",
                           alpha=0.7, s=4, label="worse than default")
                ax.scatter(x=real[same_idx],
                           y=pred[same_idx], linewidth=0.5, marker='x',
                           facecolor='', c='', edgecolor='#636363',
                           s=3, zorder=0,
                           label="medium performance")

                labelsize=20
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylabel("predicted runtime [sec]", size=labelsize)
                ax.set_xlabel("true runtime [sec]", size=labelsize)

                ax.plot([0.001, args.cutoff], [0.001, args.cutoff],
                        c='k', linewidth=0.5, zorder=89)
                ax.plot([args.cutoff, args.cutoff], [0.0001, args.cutoff],
                        c='k', linestyle=':')
                ax.plot([0.0001, args.cutoff], [args.cutoff, args.cutoff],
                        c='k', linestyle=':')
                plt.tick_params(axis='both', which='major',
                                labelsize=labelsize*0.75)
                plt.xlim([numpy.min(real)*0.9, max(numpy.max(pred), numpy.max(real)) * 1.2])
                plt.ylim([numpy.min(real)*0.9, max(numpy.max(pred), numpy.max(real)) * 1.2])

                plt.legend(markerscale=2, prop={'size': labelsize*0.75}, fancybox=True,
                           framealpha=0.5, loc="lower right")

                if args.save is not None:
                    tight_layout()
                    subplots_adjust(top=0.85)
                    save_fn = args.save + "_%s_scatter" % mode
                    logging.info("Save to %s" % save_fn)
                    plt.savefig(save_fn + ".png", dpi=dpi, facecolor='w',
                                edgecolor='w', orientation='portrait',
                                format='png', transparent=False, bbox_inches="tight",
                                pad_inches=0.1)
                else:
                    plt.show()
                plt.clf()
            else:
                    pass

        # Now build table
        num_fmt = "$%4.2f \mp %4.2f$"
        n_round = 2
        tab_dict = OrderedDict()
        tab_dict["RMSE_high"] = num_fmt % (round(numpy.mean(rmse_better), n_round),
                                           round(numpy.std(rmse_better), n_round))
        tab_dict["CC_high"] = num_fmt % (round(numpy.mean(cc_better), n_round),
                                         round(numpy.std(cc_better), n_round))
        tab_dict["RMSE_worse"] = num_fmt % (round(numpy.mean(rmse_worse), n_round),
                                            round(numpy.std(rmse_worse), n_round))
        tab_dict["CC_worse"] = num_fmt % (round(numpy.mean(cc_worse), n_round),
                                          round(numpy.std(cc_worse), n_round))
        tab_dict["RMSE_all"] = num_fmt % (round(numpy.mean(rmse_all), n_round),
                                          round(numpy.std(rmse_all), n_round))
        tab_dict["CC_all"] = num_fmt % (round(numpy.mean(cc_all), n_round),
                                        round(numpy.std(cc_all), n_round))

        for key in tab_dict.keys():
            tab_dict[key] = [tab_dict[key], ]

        a = tabulate(tab_dict, headers="keys", tablefmt="latex")
        a = a.replace('\\$', '$')
        a = a.replace(r'\textbackslash{}', '\\')
        print("")
        print(a)
if __name__ == "__main__":
    main()