#!/usr/local/bin/python2.7
# encoding: utf-8
"""
EPM -- emperical performance models

@author:     Katharina Eggensperger and Marius Lindauer

@copyright:  2015 AAD Group Freiburg. All rights reserved.

@license:    GPLv2

@contact:    {eggenspk,lindauer}@cs.uni-freiburg.de
"""

import sys
import os
import logging

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

__version__ = 0.1
__date__ = '2015-03-19'
__updated__ = '2015-03-19'


def main(argv=None):
    """Command line options."""

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by AAD on %s.
  Copyright 2015 AAD Group Freiburg. All rights reserved.

  Licensed under GPLv2
  http://www.gnu.org/licenses/gpl-2.0.html

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=ArgumentDefaultsHelpFormatter)
        req_params = parser.add_argument_group("Required")
        
        opt_params = parser.add_argument_group("Optional")
        opt_params.add_argument("-v", "--verbosity", dest="verbosity", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="verbosity level")
        opt_params.add_argument('-V', '--version', action='version', version=program_version_message)
        # Process arguments
        args = parser.parse_args()
        
        if args.verbosity == "INFO":
            logging.basicConfig(level=logging.INFO)
        elif args.verbosity == "DEBUG":
            logging.basicConfig(level=logging.DEBUG)
        elif args.verbosity == "WARNING":
            logging.basicConfig(level=logging.WARNING)
        elif args.verbosity == "ERROR":
            logging.basicConfig(level=logging.ERROR)

        logging.debug(str(args))

    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 1

if __name__ == "__main__":
    sys.exit(main())