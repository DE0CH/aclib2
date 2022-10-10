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
import filecmp
import glob
import os
import sys
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Merge validate_dirs into one dir',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='dirs', metavar='DIRS', nargs='+',
                        help='directory files to merge')
    parser.add_argument('--out', dest='out_dir', default=None, required=True,
                        help="dir to write to")
    args, unknown = parser.parse_known_args()

    assert not os.path.isdir(args.out_dir), "%s already exists" % args.out_dir
    # Get files to merge
    obj_fl = list()
    str_fl = list()
    for d in args.dirs:
        n = os.path.join(d, "validationObjectiveMatrix-cli-*-walltimeworker.csv")
        try:
            obj_fl.append(glob.glob(n)[0])
        except IndexError:
            print("Could not find %s" % n)
            raise

        n = os.path.join(d, "validationCallStrings-cli-*-walltimeworker.csv")
        try:
            str_fl.append(glob.glob(n)[0])
        except IndexError:
            raise IndexError("Could not find %s" % n)

    assert len(obj_fl) == len(str_fl) != 0

    os.mkdir(args.out_dir)

    # First copy str
    shutil.copy(str_fl[0], args.out_dir)

    # Then merge obj files
    header_written = False
    ct = -1
    with open(os.path.join(args.out_dir, os.path.basename(obj_fl[0])), 'w') \
            as outfile:
        for o_fl in obj_fl:
            with open(o_fl, 'r') as o_fh:
                if header_written:
                    o_fh.readline()
                for line in o_fh:
                    ct += 1
                    header_written = True
                    outfile.write(line)
        sys.stdout.write("Written %05d instances)\n" % ct)
