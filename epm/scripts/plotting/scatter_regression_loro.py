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
import os
from collections import OrderedDict

try:
    import cPickle as pickle
except ImportError:
    import pickle

import matplotlib.pyplot as plt
import numpy as np

import plottingscripts.plotting.scatter

import epm.experiment_utils.script_helper as script_helper
import epm.experiment_utils.data_handling as data_handling


def main():
    parser = argparse.ArgumentParser(description='Scatter test_files')
    parser.add_argument('predictions', metavar='PKL', nargs='+',
                        help='PKL files to read')
    parser.add_argument('--save', dest='save', default=None,
                        help="Where to save plots")
    parser.add_argument('--cutoff', dest='cutoff', default=None,
                        required=False, type=float,
                        help="What was the cutoff time?")
    parser.add_argument('--par', dest='par', default=10, type=int,
                        help="Treat TIMEOUTS as PARX")
    parser.add_argument('--quality', dest="quality", default=False,
                        action="store_true", help="Handle quality data?")
    args, unknown = parser.parse_known_args()

    if not args.quality and args.cutoff is None:
        print("Require --cutoff")

    rmse_dict = OrderedDict()
    cc_dict = OrderedDict()
    for fl in args.predictions:
        with open(fl, 'rb') as fh:
            test_y, pred_y = pickle.load(fh)

        plot_save = os.path.basename(fl).split(".")[0]

        if not args.quality:
            test_y[test_y >= args.cutoff] = args.cutoff*args.par
            pred_y[pred_y >= args.cutoff] = args.cutoff*args.par
            max_val = args.cutoff
            metric = "runtime"
            labels = ("True Par10 Score", "Predicted Par10 Score")
        else:
            labels = ("True quality", "Predicted quality")
            max_val = 1
            metric = "quality"

        plt.clf()
        dpi = 300
        fig = plottingscripts.plotting.scatter.plot_scatter_plot(
                x_data=test_y, y_data=pred_y,
                labels=labels, title="",
                debug=False, min_val=None, max_val=max_val, grey_factor=2,
                linefactors=[2, ], user_fontsize=20, dpi=dpi,
                metric=metric, jitter_timeout=True, markers=["x", "+", "."],
                sizes=[10, 7, 7])

        fig.savefig(args.save + ("%s.png" % plot_save),
                    dpi=dpi, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, pad_inches=0.02, bbox_inches='tight')

        pred_y = data_handling.warp(pred_y, quality=args.quality)
        test_y = data_handling.warp(test_y, quality=args.quality)

        rmse_dict[plot_save] = script_helper.wrap_rmse(y_pred=pred_y, y_true=test_y)
        cc_dict[plot_save] = script_helper.wrap_spearman(y_pred=pred_y, y_true=test_y)

    print(rmse_dict)
    print(cc_dict)

    with open("loro_regression.txt", 'w') as fh:
        fh.write("metric," + ",".join([k for k in rmse_dict]) + ";average" + "\n")
        fh.write("RMSE," + ",".join("%g" % rmse_dict[k] for k in rmse_dict) + ";%g" % np.mean(rmse_dict.values()) + "\n")
        fh.write("CC," + ",".join("%g" % cc_dict[k] for k in cc_dict) + ";%g" % np.mean(cc_dict.values()) + "\n")


if __name__ == "__main__":
    main()