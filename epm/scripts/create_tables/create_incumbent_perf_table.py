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
import glob
import logging
import os

import numpy as np
from epm.reader.output_parser import read_trajectory_file
from tabulate import tabulate

logger = logging.getLogger("create_incumbent_perf_table")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on data')
    parser.add_argument('opt_dir', metavar='OptimizerFolder', nargs='+',
                        help='Folder where run-* folders are')
    args, unknown = parser.parse_known_args()

    tab_dict = dict()
    perf_dict = dict()
    best_opt = [None, np.inf]
    for o_dir in args.opt_dir:
        o_dir = os.path.abspath(o_dir)
        optimizer = os.path.basename(o_dir)
        perf_dict[optimizer] = list()
        for i in range(1, 11):
            file = "run-%s/validate-time/validationResults-*-" \
                   "*time*.csv" % i
            file = os.path.abspath(os.path.join(o_dir, file))
            try:
                traj_file = glob.glob(file)[0]
            except IndexError:
                logging.critical("Could not find file %s" % file)
                raise

            traj_dict = read_trajectory_file(traj_file)
            perf_dict[optimizer].append(traj_dict[-1]["Test Set Performance"])

        assert len(perf_dict[optimizer]) == 10

        median_ = np.median(perf_dict[optimizer])
        if median_ <= best_opt[1]:
            best_opt = [optimizer, median_]

    logging.info("Best Optimizer is %s with %f" % (best_opt[0], best_opt[1]))

    n_round = 2
    bold_fmt = r"%4.2f, \textit{\textbf{%4.2f}}, %4.2f"
    regular_fmt = "%4.2f, %4.2f, %4.2f"

    for opt in perf_dict.keys():
        median = round(np.median(perf_dict[opt]), n_round)
        upper = round(np.percentile(a=perf_dict[opt], q=75), n_round)
        lower = round(np.percentile(a=perf_dict[opt], q=25), n_round)
        if opt == best_opt[0]:
            tab_dict[opt] = [bold_fmt % (lower, median, upper), ]
        else:
            tab_dict[opt] = [regular_fmt % (lower, median, upper), ]

    a = tabulate(tab_dict, headers="keys", tablefmt="latex")
    a = a.replace('\\$', '$')
    a = a.replace(r'\textbackslash{}', '\\')
    a = a.replace(r'\{', '{').replace(r'\}', '}')

    print(a)


if __name__ == "__main__":
    main()
