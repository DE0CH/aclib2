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
import logging
import glob
import os
from functools import partial

import numpy
import sklearn.metrics
from scipy import stats
import scipy

one_entry_template = " %s "
two_entry_template = "%4.2f $\pm$ %4.1f"
bold_one_entry_template = "\\textbf{\underline{ %4.2f }}"
bold_two_entry_template = "\\textbf{\underline{ %4.2f }} $\pm$ %4.1f"
underline_two_entry_template = "\\underline{ %4.2f } $\pm$ %4.1f"
underline_one_entry_template = "\\underline{ %4.2f }"
fail_entry_template = "\\textit{ %s }"

column_header = "\multicolumn{%d}{l}{%s}"

template_string = \
"""
\documentclass[landscape]{article} %% For LaTeX2
\usepackage[a4paper, left=5mm, right=5mm, top=5mm, bottom=5mm]{geometry}

%%\\documentclass[landscape]{article}% % For LaTeX2

%%\\usepackage[landscape]{geometry}
\\usepackage{multirow}           %% import command \multicolmun
\\usepackage{tabularx}           %% Convenient table formatting
\\usepackage{booktabs}           %% provides \\toprule, \midrule and \\bottomrule

\\begin{document}

%s

\\end{document}
"""

table_str = \
"""
\\begin{table}[t!]
\\centering
\\scriptsize
\\caption{ %s }
\\begin{tabularx}{\\textwidth}{ %s }
\\toprule
%s
\\toprule
\\\\
%s
\\\\
\\toprule
\\bottomrule
\\end{tabularx}
\\end{table}
"""

caption_content = "Not yet described"
#"Used algorithms to build surrogate models and the resulting performance on benchmarks used for a 5-fold crossvalidation on all available data. " + \
#"For each entry, bold face indicates the best performance on this benchmark, and underlined values are not statistically significantly different " + \
#"from the best according to an paired t-test (with p=0.05). " + \
#"'nan' means result could not be calculated, because algorithm crashed, 'none' means is not yet finished"


def wrap_spearman(y_true, y_pred):
    coef, _p = scipy.stats.spearmanr(a=y_pred, b=y_true)
    return coef


def wrap_rmse(y_true, y_pred):
    return numpy.sqrt(sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred))


def wrap_le(a, b):
    return a <= b


def wrap_ge(a, b):
    return a >= b


def wrap_prec(y_true, y_pred, cutoff):
    # Precision (P) is defined as the number of true positives (T_p)
    # over the number of true positives plus the number of false positives (F_p)
    tp_timeout = sum([1 if y_pred[i] >= cutoff and y_true[i] >= cutoff else 0
                      for i in range(y_true.shape[0])])
    fp_timeout = sum([1 if y_pred[i] >= cutoff and y_true[i] < cutoff else 0
                      for i in range(y_true.shape[0])])
    if tp_timeout == 0 and fp_timeout == 0:
        return -1
    else:
        return tp_timeout / (tp_timeout + fp_timeout)


def main():
    parser = argparse.ArgumentParser(description='Create table with regression results')
    parser.add_argument('-s', '--save', dest='save', default=None,
                        help="Where to save table")
    parser.add_argument('-d', dest="debug", default=False, action="store_true", help="Show logger output")
    parser.add_argument('pkl', metavar='PKL', nargs='+', help='pkl files to read')

    args, unknown = parser.parse_known_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # This is what we will analyze
    metrics = (#("MAE", sklearn.metrics.mean_absolute_error, wrap_le),
               #("MSE", sklearn.metrics.mean_squared_error, wrap_le),
               ("RMSE", wrap_rmse, wrap_le),
               #("R2", sklearn.metrics.r2_score, wrap_ge),
               ("CC", wrap_spearman, wrap_ge),
               ("Precision", wrap_prec, wrap_ge)
               #("TrainingTime", None, wrap_le)
               )
    folder_list = list()
    result_dict = dict()

    if len(args.pkl) == 1:
        args.pkl = glob.glob(args.pkl)

    # Loading data
    for pkl in args.pkl:

        # Find out cutoff for this scenario
        if "CPLEX" in pkl:
            cutoff = numpy.log10(10000)
        elif "CSSC" in pkl:
            cutoff = numpy.log10(300)
        elif "weighted" in pkl:
            cutoff = numpy.log10(900)
        elif "IBM" in pkl:
            cutoff = numpy.log10(300)
        else:
            raise ValueError("Automatic cutoff detection failed, %s" % pkl)

        logging.info("#"*80)
        logging.info("Loading Dataset %s" % pkl)


        n_data = None
        n_folds = None
        crossval_data_test = None
        cv_uncen_data_test, uc_pred_list, uc_train_time, ip_pred_list, \
            ip_train_time, data_traj, change_traj = \
            pickle.load(open(pkl, 'rb'))
        del data_traj, change_traj

        result_dict[pkl] = dict()

        for mode in ["train", "new_config", "new_inst", "both"]:
            result_dict[pkl][mode] = dict()

            # loop over all models to load data and calc metric
            logging.info("Handling mode='%s'" % mode)
            crossval_uncen_data_test = cv_uncen_data_test[mode]
            uncen_pred_list = uc_pred_list[mode]
            uncen_train_time = uc_train_time[mode]
            imp_pred_list = ip_pred_list[mode]
            imp_train_time = ip_train_time[mode]

            if n_data is None:
                n_folds = len(uncen_pred_list)
                assert len(imp_pred_list) == n_folds
                crossval_data_test = crossval_uncen_data_test
                logging.info("Found data with samples in fold 0 (%d)" % n_folds)
            else:
                logging.info("Reuse old testdata")
                assert crossval_uncen_data_test[1][1][25] == crossval_data_test[1][1][25]
                del crossval_uncen_data_test

            result_dict[pkl][mode]["UNCEN"] = dict()
            result_dict[pkl][mode]["IMP"] = dict()
            for me_name, me_func, me_comp in metrics:
                logging.info("Calculating %s" % str(me_name))
                result_dict[pkl][mode]["UNCEN"][me_name] = list()
                result_dict[pkl][mode]["IMP"][me_name] = list()

                if me_name == "Precision":
                    me_func = partial(wrap_prec, cutoff=cutoff)

                if me_name == "TrainingTime":
                    uncen_me_mean = numpy.mean(uncen_train_time)
                    uncen_me_std = numpy.std(uncen_train_time)
                    uncen_me_sig = 2
                    imp_me_mean = numpy.mean(imp_train_time)
                    imp_me_std = numpy.std(imp_train_time)
                    imp_me_sig = 2
                else:
                    for cv in range(n_folds):
                        perf = me_func(y_true=numpy.log10(crossval_data_test[cv][1]),
                                       y_pred=numpy.log10(uncen_pred_list[cv]))
                        result_dict[pkl][mode]["UNCEN"][me_name].append(perf)

                        perf = me_func(y_true=numpy.log10(crossval_data_test[cv][1]),
                                       y_pred=numpy.log10(imp_pred_list[cv]))
                        result_dict[pkl][mode]["IMP"][me_name].append(perf)
                    uncen_me_mean = round(numpy.mean(result_dict[pkl][mode]["UNCEN"][me_name]), 2)
                    uncen_me_std = round(numpy.std(result_dict[pkl][mode]["UNCEN"][me_name]), 1)
                    uncen_me_sig = 2

                    imp_me_mean = round(numpy.mean(result_dict[pkl][mode]["IMP"][me_name]), 2)
                    imp_me_std = round(numpy.std(result_dict[pkl][mode]["IMP"][me_name]), 1)
                    imp_me_sig = 2

                    logging.debug("UNCEN: %s: %f +- %f" % (me_name, uncen_me_mean, uncen_me_std))
                    logging.debug("IMPUT: %s: %f +- %f" % (me_name, imp_me_mean, imp_me_std))


                result_dict[pkl][mode]["UNCEN"][me_name + "_mean"] = uncen_me_mean
                result_dict[pkl][mode]["UNCEN"][me_name + "_std"] = uncen_me_std
                result_dict[pkl][mode]["UNCEN"][me_name + "_sig"] = uncen_me_sig

                result_dict[pkl][mode]["IMP"][me_name + "_mean"] = imp_me_mean
                result_dict[pkl][mode]["IMP"][me_name + "_std"] = imp_me_std
                result_dict[pkl][mode]["IMP"][me_name + "_sig"] = imp_me_sig
            del uc_pred_list[mode], uc_train_time[mode], ip_pred_list[mode], ip_train_time[mode]

    logging.info("%s" % str(result_dict))

    # Now get significance
    for pkl in args.pkl:
        logging.debug("Calculating significance for %s" % pkl)
        if result_dict[pkl] is None: continue
        for mode in ["train", "new_config", "new_inst", "both"]:
            logging.debug("Processing mode %s" % mode)
            for me_name, me_func, me_comp in metrics:
                if result_dict[pkl][mode]["IMP"] is None or result_dict[pkl][mode]["UNCEN"] is None: continue

                if me_comp(result_dict[pkl][mode]["UNCEN"][me_name + "_mean"], result_dict[pkl][mode]["IMP"][me_name + "_mean"]):
                    best_model = "UNCEN"
                else:
                    best_model = "IMP"

                # Now we have to repeat this loop in order to modify entries
                for t in ["UNCEN", "IMP"]:
                    print("#### %s" % t)
                    if t == best_model:
                        result_dict[pkl][mode][t][me_name + "_sig"] = 0
                    elif t != best_model:
                        t, p = stats.ttest_rel(a=result_dict[pkl][mode][t][me_name],
                                               b=result_dict[pkl][mode][t][me_name])
                        # if we have a p-value greater 0.05 the result does not differ that much
                        if p >= 0.05:
                            result_dict[pkl][mode][t][me_name + "_sig"] = 1
                    else:
                        raise ValueError("Could not find %s, %s" % (t, best_model))


    # Now create table body
    args.pkl.sort()
    table = ""
    header = ""
    body = ""

    # Create table header
    cs = 0
    num_metrics = len(metrics)
    tmp = list()
    tmp.append(column_header % (1, "Dataset"))
    tmp.append(column_header % (num_metrics*4, "non imputed"))
    #tmp.append(column_header % (num_metrics*4, "imputed"))
    header += " & ".join(tmp) + "\\\\ \n"
    header += "{} & %s & %s & %s & %s &" % (column_header % (num_metrics, "train"), column_header % (num_metrics, "new inst"), column_header % (num_metrics, "new config"), column_header % (num_metrics, "both"))
    header += "%s & %s & %s & %s  \\\\\n" % (column_header % (num_metrics, "train"), column_header % (num_metrics, "new inst"), column_header % (num_metrics, "new config"), column_header % (num_metrics, "both"))

    tmp = list(["{}", ])
    for i in range(8):
        tmp.append("RMSE")
        tmp.append("CC")
        tmp.append("TIMEOUT")
    header += " & ".join(tmp) + "\\\\ \n"
    cs = "l llllllllllll"

    header += " \\\\\n\\midrule"

    for pkl in args.pkl:
        logging.debug("Writing %s" % pkl)
        tmp = list()
        name = os.path.basename(pkl).split("_rf_")[0].replace("modes_predictions.pkl", "").replace("_", " ")
        tmp.append(name)
        for t in ["UNCEN",]:
            for mode in ["train", "new_config", "new_inst", "both"]:
                for me_name, me_func, me_comp in metrics:
                    if result_dict[pkl] is None:
                        tmp.append(one_entry_template % "None")
                        continue
                    elif result_dict[pkl][mode][t] is None:
                        # No entries for this model and experiment
                        tmp.append(one_entry_template % "None")
                        continue
                    elif result_dict[pkl][mode][t][me_name + "_mean"] >= 10000:
                        tmp.append(one_entry_template % "m$\ge 10000$")
                        continue
                    elif result_dict[pkl][mode][t][me_name + "_mean"] <= -10000:
                        tmp.append(one_entry_template % "$\le -10000$")
                        continue
                    else:
                       # We have valid entries
                        mean = result_dict[pkl][mode][t][me_name + "_mean"]
                        std = result_dict[pkl][mode][t][me_name + "_std"]

                    if result_dict[pkl][mode][t][me_name + "_sig"] == 2:
                        # regular entry
                        #tmp.append(two_entry_template % (mean, std))
                        tmp.append(one_entry_template % mean)
                    elif result_dict[pkl][mode][t][me_name + "_sig"] == 1:
                        # This is not worse
                        tmp.append(underline_one_entry_template % mean)
                    elif result_dict[pkl][mode][t][me_name + "_sig"] == 0:
                        # This is the best
                        tmp.append(bold_one_entry_template % mean)
                    else:
                        raise ValueError("Do not know significance level %d" % result_dict[pkl][mode][t][me_name + "_sig"])
        body += " & ".join(tmp)
        body += " \\\\\n"
    table1 = table_str % (caption_content, cs, header, body)

    for pkl in args.pkl:
        logging.debug("Writing %s" % pkl)
        tmp = list()
        name = os.path.basename(pkl).split("_rf_")[0].replace("modes_predictions.pkl", "").replace("_", " ")
        tmp.append(name)
        for t in ["UNCEN",]:
            for mode in ["train", "new_config", "new_inst", "both"]:
                for me_name, me_func, me_comp in metrics:
                    if result_dict[pkl] is None:
                        tmp.append(one_entry_template % "None")
                        continue
                    elif result_dict[pkl][mode][t] is None:
                        # No entries for this model and experiment
                        tmp.append(one_entry_template % "None")
                        continue
                    elif result_dict[pkl][mode][t][me_name + "_mean"] >= 10000:
                        tmp.append(one_entry_template % "m$\ge 10000$")
                        continue
                    elif result_dict[pkl][mode][t][me_name + "_mean"] <= -10000:
                        tmp.append(one_entry_template % "$\le -10000$")
                        continue
                    else:
                       # We have valid entries
                        mean = result_dict[pkl][mode][t][me_name + "_mean"]
                        std = result_dict[pkl][mode][t][me_name + "_std"]

                    if result_dict[pkl][mode][t][me_name + "_sig"] == 2:
                        # regular entry
                        #tmp.append(two_entry_template % (mean, std))
                        tmp.append(one_entry_template % mean)
                    elif result_dict[pkl][mode][t][me_name + "_sig"] == 1:
                        # This is not worse
                        tmp.append(underline_one_entry_template % mean)
                    elif result_dict[pkl][mode][t][me_name + "_sig"] == 0:
                        # This is the best
                        tmp.append(bold_one_entry_template % mean)
                    else:
                        raise ValueError("Do not know significance level %d" % result_dict[pkl][mode][t][me_name + "_sig"])
        body += " & ".join(tmp)
        body += " \\\\\n"
    table1 = table_str % (caption_content, cs, header, body)

    tex = template_string % table

    if args.save != "":
        fh = open(args.save, 'w')
        fh.write(tex)
        fh.close()
        print("Output written to %s" % args.save)
    else:
        print(tex)


if __name__ == "__main__":
    main()