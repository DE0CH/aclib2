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

import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout, figure, subplots_adjust, subplot
import matplotlib.gridspec
import matplotlib.cm
import numpy

import epm.experiment_utils.script_helper as script_helper


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on data')
    parser.add_argument('--pkl', dest='pkl', default=None,
                        help='pkl file to read from')
    parser.add_argument('--save', dest='save', default=None,
                        help="Where to save plot")
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

    for mode in ("train", "new_inst", "new_config", "both"):
        cen_min, cen_max = script_helper.\
            get_min_max(pred=imp_pred_list[mode],
                        real=crossval_uncen_data_test[mode])
        cen_max = max(args.cutoff*args.par, cen_max)
        uncen_min, uncen_max = script_helper.\
            get_min_max(pred=uncen_pred_list[mode],
                        real=crossval_uncen_data_test[mode])
        uncen_max = max(args.cutoff*args.par, uncen_max)

        dpi = 600
        gs = matplotlib.gridspec.GridSpec(3, len(crossval_uncen_data_test[mode]))
        fig = figure(1, dpi=dpi)
        #fig.suptitle(args.pkl + " mode: %s" % mode, fontsize=16)

        metrics = (("RMSE", script_helper.wrap_rmse),
                   ("CC", script_helper.wrap_spearman))
        uncen_metric_dict = collections.OrderedDict()
        imp_metric_dict = collections.OrderedDict()
        for m in metrics:
            uncen_metric_dict[m[0]] = list()
            imp_metric_dict[m[0]] = list()
        uncen_metric_dict["timeouts"] = list()
        imp_metric_dict["timeouts"] = list()

        for f in range(len(crossval_uncen_data_test[mode])):
            # First plot performance on uncensored data
            ax = subplot(gs[0, f], aspect='equal')
            ax.grid(True, linestyle='-', linewidth=1, which='major',
                    color='gray', alpha=0.5, zorder=99)

            true_values = numpy.array(copy.deepcopy(crossval_uncen_data_test[mode][f][1]))
            pred_values = copy.deepcopy(uncen_pred_list[mode][f])

            logging.debug((numpy.min(true_values), numpy.max(true_values)))
            logging.debug((numpy.min(pred_values), numpy.max(pred_values)))

            ax.scatter(true_values, pred_values, marker='.', edgecolor='', c='',
                       s=0.3, facecolor='g', zorder=49)
            # Replace values higher than cutoff
            true_values[true_values >= args.cutoff] = args.cutoff*args.par
            pred_values[pred_values >= args.cutoff] = args.cutoff*args.par
            ax.scatter(true_values, pred_values, marker='.', edgecolor='', c='',
                       s=0.5, facecolor='k', zorder=51)
            ax.scatter(true_values, pred_values, color='',
                       facecolor='Chocolate', alpha=0.3, zorder=50)

            # Calc metric
            for m_name, m_func in metrics:
                uncen_metric_dict[m_name].append(
                        m_func(y_true=numpy.log10(true_values),
                               y_pred=numpy.log10(pred_values)))

            ax.plot([0.0001, args.cutoff],
                    [0.0001, args.cutoff],
                    c='r', linewidth=0.5, zorder=89)
            ax.plot([args.cutoff, args.cutoff], [0.0001, args.cutoff],
                    c='k', linestyle=':')
            ax.plot([0.0001, args.cutoff], [args.cutoff, args.cutoff],
                    c='k', linestyle=':')

            # Find out number of correct predicted timeouts
            n_timeout = float(sum([1 if i > args.cutoff else 0
                              for i in true_values]))
            n_tp_timeout = sum([1 if pred_values[i] > args.cutoff and
                                true_values[i] > args.cutoff else 0
                                for i in range(true_values.shape[0])])
            if n_timeout == 0:
                uncen_metric_dict["timeouts"].append(1)
            else:
                uncen_metric_dict["timeouts"].append(n_tp_timeout/n_timeout)

            ax.set_title("%d points\n%5.3f | %5.3f\n%d / %d" %
                         (len(pred_values),
                          uncen_metric_dict["RMSE"][-1],
                          uncen_metric_dict["CC"][-1], n_tp_timeout,
                          n_timeout), size=8)
            ax.tick_params(labelsize=8)
            ax.set_xlabel("true", size=8)
            ax.set_ylabel("predictions", size=8)
            ax.set_ylim([uncen_min, uncen_max])
            ax.set_xlim(ax.get_ylim())
            ax.set_xscale("log")
            ax.set_yscale("log")

        for f in range(len(crossval_uncen_data_test[mode])):

            true_values = numpy.array(copy.deepcopy(crossval_uncen_data_test[mode][f][1]))
            pred_values = copy.deepcopy(imp_pred_list[mode][f])

            # Then plot performance on imputed data
            ax = subplot(gs[1, f], aspect='equal',)
            ax.grid(True, linestyle='-', linewidth=1, which='major',
                    color='gray', alpha=0.5, zorder=99)

            ax.scatter(true_values, pred_values, marker='.', edgecolor='', c='',
                       s=0.3, facecolor='g', zorder=49)
            # Replace values higher than cutoff
            true_values[true_values >= args.cutoff] = args.cutoff*args.par
            pred_values[pred_values >= args.cutoff] = args.cutoff*args.par
            ax.scatter(true_values, pred_values, color='',
                       facecolor='SlateBlue', alpha=0.3, zorder=50)
            ax.scatter(true_values, pred_values, marker='.', edgecolor='', c='',
                       s=0.5, facecolor='k', zorder=51)
            ax.plot([0.0001, args.cutoff], [0.0001, args.cutoff], c='r', linewidth=0.5,
                    zorder=89)
            ax.plot([args.cutoff, args.cutoff], [0.0001, args.cutoff],
                    c='k', linestyle=':')
            ax.plot([0.0001, args.cutoff], [args.cutoff, args.cutoff],
                    c='k', linestyle=':')

            for m_name, m_func in metrics:
                imp_metric_dict[m_name].append(
                        m_func(y_true=numpy.log10(true_values),
                               y_pred=numpy.log10(pred_values)))

            # Find out number of correct predicted timeouts
            n_timeout = float(sum([1 if i > args.cutoff else 0
                              for i in true_values]))
            n_tp_timeout = sum([1 if pred_values[i] > args.cutoff and
                                true_values[i] > args.cutoff else 0
                                for i in range(true_values.shape[0])])
            if n_timeout == 0:
                imp_metric_dict["timeouts"].append(1)
            else:
                imp_metric_dict["timeouts"].append(n_tp_timeout/n_timeout)

            ax.set_xlabel("true", size=8)
            ax.set_ylabel("predictions", size=8)
            ax.set_title("%d points\n %5.3f | %5.3f\n %d / %d" %
                         (len(imp_pred_list[mode][f]),
                          imp_metric_dict["RMSE"][-1],
                          imp_metric_dict["CC"][-1], n_tp_timeout,
                          n_timeout), size=8)
            ax.tick_params(labelsize=8)
            ax.set_ylim([cen_min, cen_max])
            ax.set_xlim(ax.get_ylim())
            ax.set_xscale("log")
            ax.set_yscale("log")

        # Plot error metrics
        ax = subplot(gs[2, 0:1])
        x = list(range(1, len(imp_metric_dict.keys())+1))
        width = 0.25
        ax.bar([i+width for i in x], [numpy.mean(imp_metric_dict[k]) for k
                                      in imp_metric_dict],
               yerr=[numpy.std(imp_metric_dict[k]) for k in imp_metric_dict],
               width=0.25, facecolor='SlateBlue', label="imp")
        ax.bar([i-width/2 for i in x], [numpy.mean(uncen_metric_dict[k]) for k
                                        in uncen_metric_dict],
               yerr=[numpy.std(uncen_metric_dict[k]) for k in
                     uncen_metric_dict], width=0.25, facecolor='Chocolate',
               label="uncen")
        plt.xticks(x, imp_metric_dict.keys())
        plt.legend(prop={'size': 5}, fancybox=True, framealpha=0.0, loc="best")
        ax.set_xlim(0.5, len(x)+1)
        ax.set_ylim(0, 1.1)

        ax.tick_params(labelsize=8)

        # Plot change over time
        ax = subplot(gs[2, 1:2])
        for fold, traj in enumerate(change_traj[mode]):
            x = list(range(len(traj)))
            ax.plot(x, traj, label="fold %d" % fold)
        ax.set_title("Change over #iterations", size=8)
        ax.legend(prop={'size': 5}, fancybox=True, framealpha=0.0,
                  loc="best")

        ax.tick_params(labelsize=8)

        # Plot imputations
        ax = subplot(gs[2, 2:len(crossval_uncen_data_test[mode])])
        cmap = plt.get_cmap("gist_heat_r")
        max_data = 100
        ax.plot([-1, max_data+1], [args.cutoff, args.cutoff], color='k',
                linestyle=":")

        for indx, data in enumerate(data_traj[mode][0]):
            x = list(range(len(data[:max_data])))
            color = (float(indx)+1)/(len(data_traj[mode][0]))
            if indx == 0:
                ax.scatter(x, data[:max_data], edgecolor="g", linewidth=0.5,
                           facecolor='g', s=3, marker='x', zorder=99)
            if indx == data_traj[mode][0].shape[0]-1:
                ax.scatter(x, data[:max_data], edgecolor="g", linewidth=0.5,
                           facecolor='', s=3, marker='_', zorder=99)

            ax.scatter(x, data[:max_data], alpha=0.5, edgecolor="",
                       facecolor=cmap(color), s=10)
        ax.set_title("Imputed values [:%d]; fold 0" % max_data, size=8)
        ax.set_xlim([0.01, len(data[:max_data]) + 0.1])
        #ax.set_ylim([0.01, 1.1*args.cutoff*args.par])
        ax.set_yscale("log")
        ax.tick_params(labelsize=8)

        if args.save is not None:
            tight_layout()
            subplots_adjust(top=0.85)
            save_fn = args.save + "_%s_scatter.png" % mode
            logging.info("Save to %s" % save_fn)
            plt.savefig(save_fn, dpi=dpi, facecolor='w',
                        edgecolor='w', orientation='portrait',
                        format=None, transparent=False, bbox_inches="tight",
                        pad_inches=0.1)
        else:
            plt.show()
        plt.clf()

if __name__ == "__main__":
    main()