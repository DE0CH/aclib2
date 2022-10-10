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
import sys
import time

import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge .json files into one file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='files', metavar='JSON', nargs='+',
                        help='.json file(s) to merge')
    parser.add_argument('--out', dest='out_file', default=None, required=True,
                        help="file to write to")
    args, unknown = parser.parse_known_args()

    filenames = args.files
    start = time.time()
    assert not os.path.exists(args.out_file), "%s already exists" % args.out_file
    with open(args.out_file, 'w') as outfile:
        all_ct = 0
        w_ct = 0
        for fname in filenames:
            logging.info("Processing %s" % fname)
            with open(fname) as infile:
                f_ct = 0
                for line in infile:
                    if not line.startswith("{") or not line.endswith("}\n"):
                        w_ct += 1
                        sys.stdout.write("WRONG: %s" % line)
                        continue

                    if all_ct % 100 == 0:
                        sys.stdout.write("\r%05d (%05d)" % (f_ct, all_ct))
                        sys.stdout.flush()
                    outfile.write(line)
                    f_ct += 1
                    all_ct += 1
                sys.stdout.write("\rWritten %05d (%05d)\n" % (f_ct, all_ct))
    dur = time.time() - start

    logging.info("#" * 80)
    logging.info("\nTook %f sec\n" % dur)
    logging.info("Collected %d runs\n" % all_ct)
    logging.info("%d of which are lost\n" % w_ct)
    logging.info("#" * 80 + "\n")
