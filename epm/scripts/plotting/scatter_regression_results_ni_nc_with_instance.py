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

import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout, subplots_adjust, subplot
import numpy
from matplotlib import rc

from epm.experiment_utils.data_handling import load_data, partition_data
from epm.reader import output_parser

logger = logging.getLogger("scatter_regression_meani_meanc")
DOWNSAMPLE = 200000
rc('text', usetex=True)


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
    parser.add_argument('target_algo_runs', metavar='JSON', nargs='+',
                        help='JSON files to read')
    parser.add_argument('--pcs', dest='pcs_file', default=None, required=True,
                        help='PCS file to read from')
    parser.add_argument('--features', dest='feature_file', required=True,
                        help='Feature file to read from')
    parser.add_argument('--instances', dest='instances', default=None,
                        help='Use instances given in that file')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help="USE SAME SEED AS FOR evaluate_regression_ni_nc.py")
    parser.add_argument('--def', dest="default_folder", required=True, type=str,
                        help="Path to SMAC validate-def folder")
    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    rs = numpy.random.RandomState(args.seed)

    # We first read smac-validate output
    val_obj_fl = os.path.join(
            args.default_folder,
            "validationObjectiveMatrix-cli-*-walltimeworker.csv")
    val_obj_fl = glob.glob(val_obj_fl)[0]
    perf_def = output_parser.\
        read_validationObjectiveMatrix_file(val_obj_fl)

    ### We have to reread data in order to know instances/configs
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

    crossval_uncen_data_test, uncen_pred_list, _, imp_pred_list, _, _, _ = pickle.load(open(args.pkl, 'rb'))

    inst_to_id = dict()
    counter = 0
    for i in inst_feat_dict.keys():
        inst_to_id[i] = counter
        counter += 1

    for grid_idx, mode in enumerate(["train", "both"]):
        logger.critical("Processing %s" % mode)
        dpi = 600

        if mode == "train":
            bool_array = ((0, 0), (0, 1), (1, 0), (1, 1))
        elif mode == "both":
            bool_array = ((1, 1), (1, 0), (0, 1), (0, 0))

        ax = subplot(aspect='equal')
        ax.grid(True, linestyle='-', linewidth=1, which='major',
                color='gray', alpha=0.5, zorder=99)

        for ls_idx, b_idx in enumerate(bool_array):
            # Copying might not be necessary, but we do it neverthelsee
            ind = copy.deepcopy(indices[b_idx[0]][b_idx[1]])
            pred = numpy.array(copy.deepcopy(imp_pred_list[mode][ls_idx]))
            real = numpy.array(copy.deepcopy(crossval_uncen_data_test[mode][ls_idx][1]))
            current_inst_ls = [inst_to_id[inst_ls[i]] for i in ind]

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
            if ls_idx == len(bool_array) - 1:
                ax.scatter(x=real, y=pred, c=current_inst_ls,
                           edgecolor="", alpha=0.6, s=4, cmap=plt.get_cmap("jet"))
            else:
                pass
                #ax.scatter(x=better_real, y=better_pred, facecolor="#2c7bb6", edgecolor="", alpha=0.6, s=4)
                #ax.scatter(x=worse_real, y=worse_pred, facecolor="#d7191c", c='', edgecolor="", alpha=0.6, s=4)
                #ax.scatter(x=same_real, y=same_pred, facecolor="#ffffbf", c='', edgecolor="", alpha=0.6, s=4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([0.001, args.cutoff], [0.001, args.cutoff],
                c='k', linewidth=0.5, zorder=89)
        ax.plot([args.cutoff, args.cutoff], [0.0001, args.cutoff],
                c='k', linestyle=':')
        ax.plot([0.0001, args.cutoff], [args.cutoff, args.cutoff],
                c='k', linestyle=':')
        plt.xlim([numpy.min(real)*0.9, max(numpy.max(pred), numpy.max(real)) * 1.1])
        plt.ylim([numpy.min(real)*0.9, max(numpy.max(pred), numpy.max(real)) * 1.1])

        if args.save is not None:
            tight_layout()
            subplots_adjust(top=0.85)
            save_fn = args.save + "_%s_instance_scatter" % mode
            logging.info("Save to %s" % save_fn)
            plt.savefig(save_fn + ".png", dpi=dpi, facecolor='w',
                        edgecolor='w', orientation='portrait',
                        format='png', transparent=False, bbox_inches="tight",
                        pad_inches=0.1)
            #plt.savefig(save_fn + ".eps", dpi=dpi, facecolor='w',
            #            edgecolor='w', orientation='portrait',
            #            format='eps', transparent=False, bbox_inches="tight",
            #            pad_inches=0.1)
        else:
            plt.show()
        plt.clf()

if __name__ == "__main__":
    main()