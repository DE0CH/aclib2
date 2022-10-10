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

import numpy
import sklearn.metrics
from scipy import stats
import scipy

one_entry_template = " %s "
two_entry_template = "%5.3f $\pm$ %5.2f"
bold_one_entry_template = "\\textbf{\underline{ %5.3f }}"
bold_two_entry_template = "\\textbf{\underline{ %5.3f }} $\pm$ %5.2f"
underline_two_entry_template = "\\underline{ %5.3f } $\pm$ %5.2f"
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


def main():
    parser = argparse.ArgumentParser(description='Create table with regression results')
    parser.add_argument('-s', '--save', dest='save', default=None,
                        help="Where to save table")
    parser.add_argument('-d', dest="debug", default=False, action="store_true", help="Show logger output")
    parser.add_argument('--cen', dest="cen", default=False, action="store_true", help="Plot table for censored data?")

    args, unknown = parser.parse_known_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # This is what we will analyze
#     models = model_dict.keys()
    models = ['rf']
    print(models)
    metrics = (("MAE", sklearn.metrics.mean_absolute_error, wrap_le),
               #("MSE", sklearn.metrics.mean_squared_error, wrap_le),
               ("RMSE", wrap_rmse, wrap_le),
               #("R2", sklearn.metrics.r2_score, wrap_ge),
               ("CC", wrap_spearman, wrap_ge),
               ("TrainingTime", None, wrap_le)
               )
    folder_list = list()
    result_dict = dict()

    if len(unknown) == 1:
        unknown = glob.glob(unknown)

    # Loading data
    for folder in unknown:
        if not os.path.isdir(folder):
            logging.info("Is not a directory: %s" % folder)

        logging.info("#"*80)
        logging.info("Loading Dataset %s" % folder)
        folder = os.path.abspath(folder)
        folder_name = os.path.basename(folder)
        folder_list.append(folder_name)

        result_dict[folder_name] = dict()

        if not os.path.isdir(folder):
            logging.info("%s does not exist" % folder)
            result_dict[folder_name] = None
            continue
        result_dict[folder_name] = dict()

        n_data = None
        n_folds = None
        crossval_data_test = None
        for m in models:
            n_data = None
            # loop over all models to load data and calc metric
            logging.info("Loading %s" % m)
            model_pkl = os.path.join(folder, folder_name + '-' + m + "_predictions.pkl")
            print(model_pkl)
            if not os.path.exists(model_pkl):
                logging.info("%s does not exist" % model_pkl)
                result_dict[folder_name][m] = None
                continue
#             crossval_uncen_data_test, uncen_pred_list, uncen_train_time, imp_pred_list, imp_train_time, data_traj, change_traj = pickle.load(file(model_pkl))
            with open(model_pkl, 'rb') as file:
                crossval_uncen_data_test, uncen_pred_list, uncen_train_time = \
                    pickle.load(file)
            imp_pred_list = uncen_pred_list
            imp_train_time = uncen_train_time
            if args.cen:
                pred_list = uncen_pred_list
                crossval_data_test_tmp = crossval_uncen_data_test
                train_time = uncen_train_time
            else:
                # Plot results on imputed data
                pred_list = imp_pred_list
                crossval_data_test_tmp = crossval_uncen_data_test
                train_time = imp_train_time
            # pred_list, crossval_data_test_tmp, train_time = pickle.load(file(model_pkl))
            if n_data == None:
                n_data = pred_list[0].shape[0]
                n_folds = len(pred_list)
                crossval_data_test = crossval_data_test_tmp
                logging.info("Found data with %d samples in fold 0 (%d)" % (n_data, n_folds))
            else:
                logging.info("Reuse old testdata")
                assert n_folds == len(pred_list)
                assert pred_list[1].shape[0] == n_data
                assert crossval_data_test[0][1].shape[0] == n_data
                assert crossval_data_test_tmp[1][1][25] == crossval_data_test[1][1][25]
            del crossval_data_test_tmp

            result_dict[folder_name][m] = dict()

            for me_name, me_func, me_comp in metrics:
                logging.info("Calculating %s" % str(me_name))
                result_dict[folder_name][m][me_name] = list()
                if me_name == "TrainingTime":
                    result_dict[folder_name][m][me_name + "_mean"] = numpy.mean(train_time)
                    result_dict[folder_name][m][me_name + "_std"] = numpy.std(train_time)
                    result_dict[folder_name][m][me_name + "_sig"] = 2
                    continue
                for cv in range(len(pred_list)):
                    perf = me_func(y_true=numpy.log(crossval_data_test[cv][1]),
                                   y_pred=numpy.log(pred_list[cv]))
                    result_dict[folder_name][m][me_name].append(perf)
                result_dict[folder_name][m][me_name + "_mean"] = numpy.mean(result_dict[folder_name][m][me_name])
                result_dict[folder_name][m][me_name + "_std"] = numpy.std(result_dict[folder_name][m][me_name])
                result_dict[folder_name][m][me_name + "_sig"] = 2
                logging.debug("%s (%s): %5.3f +- %5.3f (%s)" %
                             (m, me_name, result_dict[folder_name][m][me_name + "_mean"],
                              result_dict[folder_name][m][me_name + "_std"], str(result_dict[folder_name][m][me_name])))

            del pred_list

    logging.info("%s" % str(result_dict))

    # Now get significance
    for folder in folder_list:
        if result_dict[folder] is None: continue
        for me_name, me_func, me_comp in metrics:
            best_value = None
            best_model = None
            for m in models:
                if result_dict[folder][m] is None: continue
                if best_model is None or me_comp(result_dict[folder][m][me_name + "_mean"], best_value):
                    best_value = result_dict[folder][m][me_name + "_mean"]
                    best_model = m

            # Now we have to repeat this loop in order to modify entries
            for m in models:
                if result_dict[folder][m] is None: continue
                elif m == best_model:
                    result_dict[folder][m][me_name + "_sig"] = 0
                elif m != best_model:
                    t, p = stats.ttest_rel(a=result_dict[folder][best_model][me_name],
                                           b=result_dict[folder][m][me_name])
                    # if we have a p-value greater 0.05 the result does not differ that much
                    try:
                        if p >= 0.05:
                            result_dict[folder][m][me_name + "_sig"] = 1
                    except ValueError:
                        result_dict[folder][m][me_name + '_sig'] = 9
                            

    # Now create table body
    print(folder_list)
    folder_list.sort()
    models.sort()
    table = ""

    while len(folder_list) > 0:
        logging.debug("Left: %s" % str(folder_list))
        if len(folder_list) >= 2:
            next_exp = 2
        else:
            next_exp = len(folder_list)

        header = ""
        body = ""

        # Create table header
        cs = 0
        tmp = list()
        tmp.append(column_header % (1, "Model"))
        for e in folder_list[:next_exp]:
            print(e)
            cs += len(metrics)
            tmp.append(column_header % (len(metrics), "\\verb|" + e + "|"))
        header += " & ".join(tmp)
        header += " \\\\\n"
        cs = "l" + "".join(['l'] * cs)

        # Second row
        tmp = list()
        tmp.append(column_header % (1, ""))

        for e in folder_list[:next_exp]:
            for me_name, me_func, me_comp in metrics:
                tmp.append(column_header % (1, me_name))
        header += " & ".join(tmp)
        header += " \\\\\n\\midrule"

        #tmp = list()
        #tmp.append("\\#Data")
        #for e in folder_list[:next_exp]:
        #    tmp.append(column_header % (2, data_dict[e]['size']))
        #header += " & ".join(tmp)
        #header += " \\\\\n"

        for m in models:
            tmp = list()
            tmp.append(m)
            for folder in folder_list[:next_exp]:
                for me_name, me_func, me_comp in metrics:
                    if result_dict[folder] is None:
                        tmp.append(one_entry_template % "None")
                        continue
                    elif result_dict[folder][m] is None:
                        # No entries for this model and experiment
                        tmp.append(one_entry_template % "None")
                        continue
                    elif result_dict[folder][m][me_name + "_mean"] >= 10000:
                        tmp.append(one_entry_template % "m$\ge 10000$")
                        continue
                    elif result_dict[folder][m][me_name + "_mean"] <= -10000:
                        tmp.append(one_entry_template % "$\le -10000$")
                        continue
                    else:
                        pass

                    # We have valid entries
                    mean = result_dict[folder][m][me_name + "_mean"]
                    std = result_dict[folder][m][me_name + "_std"]

                    if result_dict[folder][m][me_name + "_sig"] == 1:
                        # This is not worse
                        tmp.append(underline_two_entry_template % (mean, std))
                    elif result_dict[folder][m][me_name + "_sig"] == 0:
                        # This is the best
                        tmp.append(bold_two_entry_template % (mean, std))
                    else:
                        tmp.append(two_entry_template % (mean, std))

            body += " & ".join(tmp)
            body += " \\\\\n"
        table += table_str % (caption_content, cs, header, body)
        del folder_list[:next_exp]

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
