import argparse
from collections import OrderedDict
import logging
import os
import time

import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout, figure, subplots_adjust
import numpy

import epm.reader.feature_parser
import epm.reader.performance_parser
import epm.reader.instance_parser

import hashlib


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("scripts.plotting.plot_data")


def create_hash(config_dict):
    # build a string and then hash it
    string = "".join(["%s%s" % (key.strip(), config_dict[key]) for key in sorted(config_dict)])
    h = hashlib.md5(string)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Plot distribution of configs '
                                                 'across instances',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--instances', dest="instances", default=None,
                        help="File with one instance per line to use, if None "
                             "all instances in the feature file will be used")
    parser.add_argument('target_algo_runs', metavar='JSON', nargs='+',
                        help='JSON files to read')
    parser.add_argument('-s', '--save', dest='save', default=None,
                        help="Where to save predictions")
    parser.add_argument('-m', '--max_y', dest='max_y', default=300, type=float,
                        help="What is the timelimit?")
    parser.add_argument('-v', dest="verbose", default=False,
                        action='store_true', help='Show logging output')
    parser.add_argument('-d', dest="downsample", default=None, type=int,
                        help='Downsample data and use this as seed')
    parser.add_argument('--quality', dest="quality", default=False,
                        action="store_true", help='Handle performance data')
    args, unknown = parser.parse_known_args()
    numpy.random.seed(1)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for fl in args.target_algo_runs:
        if not os.path.isfile(fl):
            raise ValueError("%s is not a file" % fl)

    # Read in performance data
    logger.info("Load csv: \n%s" % "\n".join(args.target_algo_runs))
    start = time.time()
    pp = epm.reader.performance_parser.PerformanceParser(debug=args.verbose,
                                                         cutoff=args.max_y,
                                                         quality=args.quality)
    if args.quality:
        c_list, perf_list, i_list = pp.read_data_from_multiple_files(
                args.target_algo_runs)
    else:
        c_list, perf_list, i_list, to_list, cen_list = \
            pp.read_data_from_multiple_files(args.target_algo_runs)
        logger.info("% 7d (% 3g) censored runs found" % (sum(cen_list), sum(cen_list)/float(len(perf_list))))
        logger.info("% 7d (% 3g) timeout runs found" % (sum(to_list), sum(to_list)/float(len(perf_list))))
    logger.info("% 7d runs found" % len(perf_list))

    del pp

    if args.downsample is not None and args.downsample < len(i_list):
        logger.critical("DOWNSAMPLE to %d" % args.downsample)
        rs = numpy.random.RandomState(args.downsample)
        idx = list(range(len(i_list)))
        rs.shuffle(idx)
        idx = idx[:args.downsample]
        inst_list = [i_list[i] for i in idx]
        del i_list
        if not args.quality:
            cen_list = [cen_list[i] for i in idx]
        perf_list = [perf_list[i] for i in idx]
        config_list = [c_list[i] for i in idx]
        del c_list
    else:
        inst_list = i_list
        config_list = c_list

    # Read in instances
    sorted_inst_dict = OrderedDict()
    logging.info("Load instances: %s" % args.instances)
    i_start = time.time()
    ct = 0
    if args.instances is None:
        logging.info("No instances provided, use all")
        # We have to use these instances as they do not contain test instances.
        for key in inst_list:
            if key in sorted_inst_dict:
                continue
            sorted_inst_dict[key] = ct
            ct += 1
    elif args.instances is not None and not os.path.isfile(args.instances):
        raise ValueError("%s is not a file" % args.instances)
    else:
        logger.info("Read instances from %s" % args.instances)
        inst_parser = epm.reader.instance_parser.InstanceParser()
        train_inst_list = inst_parser.read_instances(args.instances)
        for ct, inst in enumerate(sorted(train_inst_list)):
            if inst in sorted_inst_dict:
                logger.critical("%s appears more than once" % inst)
            inst = inst.strip()
            sorted_inst_dict[inst] = ct
        del inst_parser, train_inst_list
        logger.info("Found %d instances, performance data is on %d instances" %
                    (ct+1, len(sorted_inst_dict.keys())))
        dur = time.time() - i_start
        logger.info("Reading instances took %ssec" % dur)

    dur = time.time() - start
    logger.info("Reading performance data took %ssec" % dur)

    # This is hacky
    logging.info("Start building matrix")
    start = time.time()
    config_idx_dict = OrderedDict()
    uncen_config_instance_list = list()
    config_set = set()
    cen_config_instance_list = list()

    test_inst_ct = 0
    all_ct = 0
    cen_ct = 0
    for idx, config in enumerate(config_list):
        # we want one hash for each config
        key = create_hash(config)
        if key not in config_idx_dict:
            # We have not seen this config
            config_idx_dict[key] = len(config_idx_dict)
            # This config gets a unique id
            uncen_config_instance_list.append(list())
            cen_config_instance_list.append(list())
        if inst_list[idx] not in sorted_inst_dict:
            if args.instances is None:
                raise ValueError("%s is not in instance list" % inst_list[idx])
            else:
                logger.debug("IGNORE: %s" % inst_list[idx])
                test_inst_ct += 1
                continue
        all_ct += 1
        if args.quality:
            # add all samples as uncen samples
            uncen_config_instance_list[config_idx_dict[key]].\
                append(sorted_inst_dict[inst_list[idx]])
        elif not cen_list[idx]:
            # add all uncen samples for this instance
            uncen_config_instance_list[config_idx_dict[key]].\
                append(sorted_inst_dict[inst_list[idx]])
        else:
            cen_ct += 1
            cen_config_instance_list[config_idx_dict[key]].\
                append(sorted_inst_dict[inst_list[idx]])
        #if inst_list[idx] not in inst_idx_dict:
        #    inst_idx_dict[inst_list[idx]] = inst_ct
        #    inst_ct += 1
        #if suc_list[idx]:
        #    uncen_config_instance_dict[key].append(inst_idx_dict[inst_list[idx]])
        #else:
        #    cen_config_instance_dict[key].append(inst_idx_dict[inst_list[idx]])
        config_set.add(key)
    dur = time.time() - start
    logger.info("Building matrix took %ssec" % dur)
    logger.info("%d/%d are evaluations on test instances and will not be "
                "plotted" % (test_inst_ct, all_ct))

    # Maybe there is a better way I am not aware of
    len_list = [(len(uncen_config_instance_list[config_idx_dict[config]]) +
                 len(cen_config_instance_list[config_idx_dict[config]]))
                for config in config_idx_dict]
    sorted_idx = reversed(numpy.argsort(len_list))

    max_x = len(sorted_inst_dict)
    max_y = len(config_set)

    dpi = 600
    fig = figure(1, dpi=dpi)
    fig.suptitle("%d configuration/%d instances/%d datapoints" %
                 (len(config_idx_dict), len(sorted_inst_dict),
                  all_ct-test_inst_ct), fontsize=16)

    del config_list, inst_list, sorted_inst_dict
    del config_idx_dict, len_list

    # build arrays to plot
    cen_x = list()
    cen_y = list()
    uncen_x = list()
    uncen_y = list()
    p_l = list()
    for i, idx in enumerate(sorted_idx):
        y = [i] * len(uncen_config_instance_list[idx])
        x = uncen_config_instance_list[idx]
        uncen_x.extend(x)
        uncen_y.extend(y)

        y = [i] * len(cen_config_instance_list[idx])
        x = cen_config_instance_list[idx]
        cen_x.extend(x)
        cen_y.extend(y)
    if args.quality:
        assert len(cen_x) == 0
    del uncen_config_instance_list, cen_config_instance_list, sorted_idx

    plt.scatter(uncen_x, uncen_y, marker="s", c='k', s=0.9, lw=0,
                label="%d uncensored samples" % (all_ct-cen_ct-test_inst_ct))
    if len(cen_x) >= 0:
        plt.scatter(cen_x, cen_y, marker="s", c='r', s=0.9, lw=0,
                    label="%d censored samples" % cen_ct)

    plt.legend(fancybox=True, loc="best", markerscale=2)
    plt.ylim([-max_y*0.1, max_y+max_y*0.1])
    plt.xlim([-max_x*0.1, max_x+max_x*0.1])
    tight_layout()
    subplots_adjust(top=0.85)
    logger.info("Save to %s" % args.save + "_scatter.png")
    plt.savefig(args.save + "_scatter.png", dpi=dpi, facecolor='w',
                edgecolor='w', orientation='portrait', papertype=None,
                format=None, transparent=False, bbox_inches="tight",
                pad_inches=0.1)

    if args.quality:
        plt.clf()
        dpi = 600
        fig = figure(1, dpi=dpi)
        mean = numpy.mean(perf_list)
        std = numpy.std(perf_list)
        median = numpy.median(perf_list)
        lower = numpy.percentile(perf_list, 25)
        upper = numpy.percentile(perf_list, 75)
        fig.suptitle("mean %g +- %g; %g || %g || %g" %
                     (mean, std, lower, median, upper))
        plt.hist(perf_list, normed=True, stacked=True)
        lim = plt.axes().get_ylim()
        plt.plot([mean, mean], [-1, 100], c='g', linewidth=3, label="mean")
        plt.fill_between([mean-std, mean+std], [-1, -1], [100, 100],
                         facecolor='g', alpha=0.2)
        plt.plot([median, median], [-1, 100], c='orange', linewidth=3,
                 label="median")
        plt.fill_between([lower, upper], [-1, -1], [100, 100],
                         facecolor='orange', alpha=0.2)
        plt.legend(fancybox=True, loc="best", markerscale=2)
        plt.ylim([0, lim[1]])
        tight_layout()
        subplots_adjust(top=0.85)
        logger.info("Save to %s" % args.save + "_histogram.png")
        plt.savefig(args.save + "histogram.png", dpi=dpi, facecolor='w',
                    edgecolor='w', orientation='portrait', papertype=None,
                    format=None, transparent=False, bbox_inches="tight",
                    pad_inches=0.1)

if __name__ == "__main__":
    main()
