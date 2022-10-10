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
import logging
import os
import shutil
import sys

logger = logging.getLogger("create_surrogate_scenario")

sync_virtualenv = """
HERE=`pwd`

mkdir -p "/tmp/eggenspk/"
LOCKDIR="/tmp/eggenspk/lock/"

for i in {1..60}; do
  if mkdir "$LOCKDIR"; then
    echo "Locking succeeded - Going to copy" >&2

    cp /home/eggenspk/ACLIB_2016/virtualEPMSync.tar.gz /tmp/eggenspk/ -f
    cd /tmp/eggenspk/
    mkdir /tmp/eggenspk/$$
    tar -xf virtualEPMSync.tar.gz -C /tmp/eggenspk/$$
    # mv /tmp/eggenspk/$$/virtualEPMSync /tmp/eggenspk/
    rsync -a /tmp/eggenspk/$$/virtualEPMSync /tmp/eggenspk/
    rm  /tmp/eggenspk/$$ -r
    break

  else
    echo "Lock failed - sleep and try again" >&2
    pid=$$
    sleeptime=$((pid % 60))

    sleep $((sleeptime+10))s
  fi
done

source /tmp/eggenspk/virtualEPMSync/bin/activate
rmdir "$LOCKDIR"

cd $HERE

"""

cmd_config_template = """
python %s/aclib/run.py -s %s -c %s --env meta --startup %s -n 10
"""

pbs_validate_template = """
python %s/aclib/validate.py -s %s -c %s --env meta --startup %s -n 10 --job_cutoff 300 --set TEST --mode TIME --num_validation_runs 10 --num_func_evals
"""


def modify_scenario(scenario_file, model_pkl):
    # Read in current File, modify and output to new place
    algo = False
    time = False
    pcs = False
    pcs_file = None
    algo_exec = None

    new_fl = open(scenario_file + ".new", "w")
    with open(scenario_file) as fl:
        for line in fl:
            if not algo and "algo" in line:
                new_fl.write("algo = python ./target_algorithms/surrogate/wrapper.py --quality 1 --model %s\n" % model_pkl)
                algo = True
                algo_exec = line.strip().split("=")[1]
            #elif not time and "wallclock_limit" in line:
            #    new_fl.write(line)
            #    limit = line.strip().split("=")[1]
            #    limit = int(float(limit.strip()))
            #    time = True
            #    new_fl.write("tunerTimeout = %d\n" % limit)
            elif not pcs and "paramfile" in line:
                new_fl.write(line)
                line = line.strip().split("=")[1]
                pcs_file = line.strip()
                pcs = True
            else:
                new_fl.write(line)
    new_fl.close()

    if not algo:
        logger.critical("Could not find 'algo' in %s" % scenario_file)
        sys.exit(1)

    #if not time:
    #    logger.critical("Could not find 'wallclock-limit' in %s" % scenario_file)
    #    sys.exit(1)

    # Then overwrite
    shutil.move(scenario_file, scenario_file + ".original")
    shutil.move(scenario_file + ".new", scenario_file)

    return pcs_file, algo_exec, os.path.abspath(scenario_file)


def create_pbs_file(scenario_name, configurator, aclib_root, model, save, startup):
    assert configurator in ("SMAC2", "PARAMILS", "ROAR", "IRACE2")
    print(scenario_name, model, configurator, aclib_root)

    # python %s/aclib/run.py -s %s -c %s --env meta --startup %s -n 10
    pbs = cmd_config_template % (aclib_root, scenario_name, configurator,
                                 startup)
    fh = open(save, "w")
    fh.write(pbs)
    fh.close()
    return


def create_validate_daemon_pbs_file(configurator, aclib_root, scenario_name,
                                    save, startup):
    # python %s/aclib/validate.py -s %s -c %s --env meta --startup %s -n 10
    #                             --job_cutoff 300 --set TEST --mode TIME
    #                             --num_validation_runs 10

    pbs = pbs_validate_template % (aclib_root, scenario_name, configurator,
                                   startup)
    fh = open(save, "w")
    fh.write(pbs)
    fh.close()
    return


def main():
    parser = argparse.ArgumentParser(description='create a new folder to run a'
                                                 ' scenario')
    parser.add_argument('--scenario', dest='scenario', required=True,
                        help='Name of scenario')
    parser.add_argument('--scenarioFile', dest='scenario_file', required=True,
                        help='Scenario file to use')
    parser.add_argument('--model', dest='model', required=True,
                        help="Where is the pickled model?")
    parser.add_argument('--save', dest='save', required=True,
                        help="Where to create Folder?")
    parser.add_argument('--aclib', dest='aclib_root', required=True,
                        help="Which aclib root to use?")
    parser.add_argument('--debug', dest="debug", default=False,
                        action="store_true", help="Show some more output")
    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # First do some checks
    if not os.path.isdir(args.save):
        logger.error("%s does not exist" % args.save)
        sys.exit(1)
    else:
        args.save = os.path.abspath(args.save)

    if not os.path.isfile(args.scenario_file):
        logging.error("%s does not exist" % args.scenario_file)
        sys.exit(1)
    else:
        args.scenario_file = os.path.abspath(args.scenario_file)

    if not os.path.isfile(args.model):
        logging.error("%s does not exist" % args.model)
        sys.exit(1)
    else:
        args.model = os.path.abspath(args.model)

    if not os.path.isdir(args.aclib_root):
        logger.error("%s does not exist" % args.aclib_root)
        sys.exit(1)
    else:
        args.aclib_root = os.path.abspath(args.aclib_root)

    # Create directory
    new_scenario_name = args.scenario + "_surrogate"
    new_scenario_dir = os.path.join(args.save, new_scenario_name)
    if os.path.isdir(new_scenario_dir):
        logger.error("%s already exists" % new_scenario_dir)
        sys.exit(1)
    else:
        os.mkdir(new_scenario_dir)

    # CP scenario file
    shutil.copy(args.scenario_file, os.path.join(new_scenario_dir,
                                                 "scenario.txt"))

    # Modify scenario file
    pcs_file, algo_exec, scenario_file = modify_scenario(
            scenario_file=os.path.join(new_scenario_dir, "scenario.txt"),
            model_pkl=args.model)

    if not os.path.isfile("./startup.sh"):
        logger.critical("Create startup.sh")
        with open("./startup.sh", 'w') as fh:
            fh.write(sync_virtualenv)
    startup = os.path.abspath(os.path.join("./startup.sh"))
    logger.critical("Startup file created in %s" % startup)

    # Create pbs File for ParamILS, SMAC
    for conf in ["SMAC2", "PARAMILS", "ROAR", "IRACE2"]:
        create_pbs_file(scenario_name=new_scenario_name, configurator=conf,
                        aclib_root=args.aclib_root, model=args.model,
                        save=os.path.join(args.save, new_scenario_name,
                                          ("run%s.pbs" % conf).replace(" ", "_")),
                        startup=startup)
        create_validate_daemon_pbs_file(configurator=conf,
                                        aclib_root=args.aclib_root,
                                        scenario_name=new_scenario_name,
                                        save=os.path.join(args.save,
                                                          new_scenario_name,
                                                          ("validateDaemon%s.pbs" % conf)).replace(" ", "_"),
                                        startup=startup)

    logger.critical("Copy the scenario file from %s" %
                    (os.path.join(new_scenario_dir, "scenario.txt")))

if __name__ == "__main__":
    main()