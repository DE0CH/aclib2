import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import glob
import logging
import os
from collections import OrderedDict

import numpy
import sys
from tabulate import tabulate

from epm.models import external_rfr_quantile
from epm.reader import output_parser
from epm.experiment_utils.script_helper import wrap_rmse, wrap_spearman

import matplotlib.pyplot as plt

logger = logging.getLogger("evaluate_surrogate_test_instances_variance")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on '
                                                 'configurations/instances'
                                                 ' evaluated for smac-validate')
    parser.add_argument('optimizer_folder', metavar='OPT', nargs='+',
                        help='Path to SMAC folder')
    parser.add_argument('--pkl', dest='model_pkl', default=None, required=True,
                        help='Model file to use')
    parser.add_argument('--save', dest='save', default=None, required=True,
                        help="Where to save Plot")
    parser.add_argument('--cutoff', dest='cutoff', default=0, type=int,
                        help="All runtimes >= cutoff are timeouts")
    parser.add_argument('--par', dest='par', default=10, type=int,
                        help="Treat TIMEOUTS as PARX")
    parser.add_argument('--dpi', dest='dpi', default=300, type=int,
                        help="dpi to use for saving plot")
    parser.add_argument('--imputation', dest="imputation", default="def",
                        help="How to impute nonactive parameter")

    args, unknown = parser.parse_known_args()

    if args.imputation == "outlier":
            args.save += "outlier_"

    rs = numpy.random.RandomState(1)

    logger.setLevel(logging.DEBUG)
    logger.info("Load model from %s" % args.model_pkl)
    model, inst_feat_dict, cs = pickle.load(open(args.model_pkl, 'rb'))

    if not isinstance(model, external_rfr_quantile.ExternalRFRQuantile):
        raise ValueError("%s is not a QF instance" % model)

    inst_list = list()
    config_list = list()
    perf_list = list()
    default_idx = list()
    incumbent_idx = list()

    for o_folder in args.optimizer_folder:
        for num_run in range(1, 11):
            val_obj_fl = os.path.join(o_folder, "run-%d" % num_run, "validate-time", "validationObjectiveMatrix-traj-run-%d-tunertimelocal.csv" % num_run)
            val_str_fl = os.path.join(o_folder, "run-%d" % num_run, "validate-time", "validationCallStrings-traj-run-%d-tunertimelocal.csv" % num_run)

            logger.info("Reading: %s" % val_obj_fl)
            if not os.path.exists(val_obj_fl):
                logger.critical("Assuming ParamILS run")
                val_obj_fl = glob.glob(os.path.join(o_folder, "run-%d" % num_run, "validate-time", "validationObjectiveMatrix-focused-runs*-traj_%d-tunertimelocal.csv" % num_run))[0]
                val_str_fl = glob.glob(os.path.join(o_folder, "run-%d" % num_run, "validate-time", "validationCallStrings-focused-runs*-traj_%d-tunertimelocal.csv" % num_run))[0]

            assert os.path.exists(val_obj_fl)
            assert os.path.exists(val_str_fl), val_str_fl

            # We first read all perfs and call_str
            perfs = output_parser.read_validationObjectiveMatrix_file(val_obj_fl)
            call_str = output_parser.read_validationCallStrings_file(val_str_fl)

            tmp_config_list = list()
            for call in call_str:
                config_vec = cs.convert_param_dict(call)
                config_vec = cs.impute_non_active(config_vec,
                                                  value=args.imputation)
                tmp_config_list.append(config_vec)

            for inst in perfs.keys():
                for idx, performance in enumerate(perfs[inst]):
                    inst_list.append(inst)
                    perf_list.append(performance)
                    feats = inst_feat_dict[inst]
                    config_list.append(numpy.hstack([tmp_config_list[idx], feats]))
                    if idx == 0:
                        default_idx.append(True)
                        incumbent_idx.append(False)
                    elif idx == len(perfs[inst])-1:
                        incumbent_idx.append(True)
                        default_idx.append(False)
                    else:
                        default_idx.append(False)
                        incumbent_idx.append(False)

    config_list = numpy.array(config_list).astype(numpy.float32)
    default_idx = numpy.array(default_idx, dtype=bool)
    incumbent_idx = numpy.array(incumbent_idx, dtype=bool)

    logger.info("Found %s config/inst pairs" % str(config_list.shape))

    # Replace all TIMEOUTS with PARX value
    perf_list = numpy.array(perf_list)
    perf_list[perf_list >= args.cutoff] = args.cutoff*args.par
    _min = 0.1 #numpy.min(perf_list)*0.9

    # Log10 data
    perf_list = numpy.log10(perf_list)

    # Predict with model
    lower = model._predict_quantile_direct(config_list, quantile=0.25)
    upper = model._predict_quantile_direct(config_list, quantile=0.75)
    pred = model.predict(config_list, seed=-1, num_samples=1)

    logger.info("Highest Prediction %g" % numpy.power(10, numpy.max(pred)))
    logger.info("Lowest Prediction %g" % numpy.power(10, numpy.min(pred)))

    logger.info("Highest real value %g" % numpy.power(10, numpy.max(perf_list)))
    logger.info("Lowest real value %g" % numpy.power(10, numpy.min(perf_list)))

    # Replace all values higher than cutoff with cutoff*par
    pred[pred >= numpy.log10(args.cutoff)] = numpy.log10(args.cutoff*args.par)
    lower[lower >= numpy.log10(args.cutoff)] = numpy.log10(args.cutoff*args.par)
    upper[upper >= numpy.log10(args.cutoff)] = numpy.log10(args.cutoff*args.par)

    # Output a table
    num_fmt = "$%4.2f$"
    n_round = 2
    tab_dict = OrderedDict()

    tab_dict["RMSE_all"] = wrap_rmse(y_true=perf_list, y_pred=pred)
    tab_dict["CC_all"] = wrap_spearman(y_true=perf_list, y_pred=pred)

    if numpy.sum(default_idx) > 0:
        tab_dict["RMSE_default"] = wrap_rmse(y_true=perf_list[default_idx],
                                             y_pred=pred[default_idx])
        tab_dict["CC_default"] = wrap_spearman(y_true=perf_list[default_idx],
                                               y_pred=pred[default_idx])
    else:
        logging.critical("No default found")
        sys.exit(1)

    if numpy.sum(incumbent_idx) > 0:
        tab_dict["RMSE_incumbent"] = wrap_rmse(
                y_true=perf_list[incumbent_idx],
                y_pred=pred[incumbent_idx])
        tab_dict["CC_incumbent"] = wrap_spearman(
                y_true=perf_list[incumbent_idx],
                y_pred=pred[incumbent_idx])
    else:
        logging.critical("No incumbent found, use nan")
        tab_dict["RMSE_incumbent"] = numpy.nan
        tab_dict["CC_incumbent"] = numpy.nan

    for key in tab_dict.keys():
            tab_dict[key] = [num_fmt % round(tab_dict[key], n_round), ]

    a = tabulate(tab_dict, headers="keys", tablefmt="latex")
    a = a.replace('\\$', '$')
    a = a.replace(r'\textbackslash{}', '\\')

    with open(args.save + "_test_instances_quantiles_RMSE.csv", 'w') as fl:
        fl.write(a)

    # unlog data
    pred = numpy.power(10, pred)
    lower = numpy.power(10, lower)
    upper = numpy.power(10, upper)

    print("MEAN PREDICTION: %f" % numpy.mean(pred[default_idx]))

    lower[lower < 0.1] = 0.1

    # Plot only 1000 points
    plot_idx = list(range(0, pred.shape[0]))
    rs.shuffle(plot_idx)
    plot_idx = plot_idx[:1000]

    print("DOWNSAMPLE TO %d points" % len(plot_idx))

    print(sum(default_idx.astype(numpy.int)),
          sum(incumbent_idx.astype(numpy.int)))
    print(default_idx.shape, incumbent_idx.shape, perf_list.shape, pred.shape,
          lower.shape, upper.shape)

    perf_list = numpy.power(10, perf_list[plot_idx])

    pred = pred[plot_idx]
    lower = lower[plot_idx]
    upper = upper[plot_idx]

    # Now we have to trick as matplotlibs errorbar function want errors relative
    # to Median
    lower = numpy.abs(lower-pred)
    upper = numpy.abs(upper-pred)

    default_idx = default_idx[plot_idx]
    incumbent_idx = incumbent_idx[plot_idx]

    print(default_idx.shape, incumbent_idx.shape, perf_list.shape, pred.shape,
          lower.shape, upper.shape)

    ax = plt.subplot(aspect='equal')
    ax.grid(True, linestyle='-', linewidth=1, which='major',
            color='gray', alpha=0.4, zorder=99)

    ax.scatter(x=perf_list[default_idx], y=pred[default_idx],
               facecolor="#d7191c", c='', edgecolor="", alpha=0.8, s=4,
               label="default")
    ax.errorbar(x=perf_list[default_idx], y=pred[default_idx],
                yerr=[lower[default_idx], upper[default_idx]], fmt='none',
                ecolor='#d7191c', alpha=0.4, markersize=4, edgecolor='',
                marker='o', capthick=0.3, elinewidth=0.3)

    ax.scatter(x=perf_list[incumbent_idx], y=pred[incumbent_idx],
               facecolor="#2c7bb6", c='', edgecolor="", alpha=0.8, s=4,
               label="final incumbent")
    ax.errorbar(x=perf_list[incumbent_idx], y=pred[incumbent_idx],
                yerr=[lower[incumbent_idx], upper[incumbent_idx]], fmt='none',
                ecolor='#2c7bb6', alpha=0.4, markersize=4, edgecolor='',
                marker='o', capthick=0.3, elinewidth=0.3)

    ax.scatter(x=perf_list, y=pred,
               linewidth=0.5, marker='x', facecolor='', c='',
               edgecolor='#404040', s=3, zorder=0,
               label="remaining samples")

    ax.legend(prop={'size': 10}, fancybox=True, framealpha=0.5, loc="best",
              markerscale=2)

    ax.plot([0.0001, args.cutoff*args.par], [0.0001, args.cutoff*args.par],
            c='r', linewidth=0.5, zorder=89)
    ax.plot([args.cutoff, args.cutoff], [0.0001, args.cutoff],
            c='k', linestyle=':')
    ax.plot([0.0001, args.cutoff], [args.cutoff, args.cutoff],
            c='k', linestyle=':')

    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("predicted runtime [sec]")
    plt.xlabel("real runtime [sec]")
    plt.xlim([_min, args.cutoff*args.par*1.2])
    plt.ylim([_min, args.cutoff*args.par*1.2])
    plt.tick_params(labelsize=8)

    mode = "png"
    if args.save is not None:
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        save_fn = args.save + "_test_instances_quantiles.%s" % mode
        logging.info("Save to %s" % save_fn)
        plt.savefig(save_fn, dpi=args.dpi, facecolor='w',
                    edgecolor='w', orientation='portrait',
                    format=None, transparent=False, bbox_inches="tight",
                    pad_inches=0.1)
    else:
        plt.show()

if __name__ == "__main__":
    main()