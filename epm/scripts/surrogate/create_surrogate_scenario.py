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
"""

pbs_template = """
#!/bin/bash
#$ -S /bin/bash
#$ -o %s/sgeout -e %s/sgeout
#$ -cwd
#$ -t 1-10

echo "Here's what we know from the SGE environment"
echo SHELL=$SHELL
echo HOME=$HOME
echo USER=$USER
echo JOB_ID=$JOB_ID
echo JOB_NAME=$JOB_NAME
echo HOSTNAME=$HOSTNAME
echo SGE_TASK_ID=$SGE_TASK_ID
echo `pwd`

HERE=`pwd`

%s

cd $HERE

echo "Job started at: `date`"

EXPERIMENT="%s"
MODEL="%s"
OPTIMIZER="%s"
OPTIMIZER_STR=${OPTIMIZER// /_}
JSON=%s

export PORTFILE=$TMPDIR/port.file

echo "Start Daemon at: `date`"
surrogate_daemonizer.py --pkl ${MODEL} --dir $TMPDIR/ --pid ${JOB_ID}_${SGE_TASK_ID} --start

echo "Sleep for 60s at: `date`"
sleep 60

echo "Check daemon at: `date`"
surrogate_daemonizer.py --pkl ${MODEL} --dir $TMPDIR/ --pid ${JOB_ID}_${SGE_TASK_ID} --status

echo "Start SMAC at: `date`"
python %s/src/run_scenario.py -C ${JSON} -d ./${EXPERIMENT}/${OPTIMIZER_STR}/run-${SGE_TASK_ID} -S ${SGE_TASK_ID} -s ${EXPERIMENT} -c ${OPTIMIZER}

echo "Terminate daemon at: `date`"
surrogate_daemonizer.py --pkl ${MODEL} --dir $TMPDIR/ --pid ${JOB_ID}_${SGE_TASK_ID} --stop

echo "Job finished with exit code $? at: `date`"
"""

pbs_validate_template = """
#!/bin/bash
#$ -S /bin/bash
#$ -o %s/sgeout -e %s/sgeout
#$ -cwd
#$ -t 1-10

echo "Here's what we know from the SGE environment"
echo SHELL=$SHELL
echo HOME=$HOME
echo USER=$USER
echo JOB_ID=$JOB_ID
echo JOB_NAME=$JOB_NAME
echo HOSTNAME=$HOSTNAME
echo SGE_TASK_ID=$SGE_TASK_ID
echo `pwd`

%s

EXPERIMENT="%s"
OPTIMIZER="%s"
OPTIMIZER=${OPTIMIZER// /_}

cd %s/${EXPERIMENT}/${OPTIMIZER}/run-${SGE_TASK_ID}
echo `pwd`

export MODELPKL=%s

ACLIB_ROOT=%s
ACLIB_VALIDATE_DIR=%s
TRAJ_FILE=%s
MYSQL_POOL=ACLIB_NIPS_surrogate
ALGO_EXEC="%s"

TAE="%s"

VALIDATE_PARAMS="--max-timestamp -1.0 --min-timestamp 0.0 --mult-factor 2.0 --validateOnlyLastIncumbent false "
PATH_PARAMS="--trajectoryFile ${TRAJ_FILE} --useScenarioOutDir true --outputDirectory ./validate-time --scenarioFile ./scenarios/*/${EXPERIMENT}/scenario.txt --algo-exec-dir ${ACLIB_VALIDATE_DIR}"

${ACLIB_ROOT}/configurators/smac-v2.10.02-master-773/smac-validate ${VALIDATE_PARAMS} ${PATH_PARAMS} ${TAE} --numRun ${SGE_TASK_ID} --algo-exec "${ALGO_EXEC}"

echo "Job finished with exit code $? at: `date`"
"""

pbs_daemon_template = """
#!/bin/bash
#$ -S /bin/bash
#$ -o %s/sgeout -e %s/sgeout
#$ -cwd
#$ -t 1-10

echo "Here's what we know from the SGE environment"
echo SHELL=$SHELL
echo HOME=$HOME
echo USER=$USER
echo JOB_ID=$JOB_ID
echo JOB_NAME=$JOB_NAME
echo HOSTNAME=$HOSTNAME
echo SGE_TASK_ID=$SGE_TASK_ID
echo `pwd`

%s

EXPERIMENT="%s"
OPTIMIZER="%s"
OPTIMIZER=${OPTIMIZER// /_}

cd %s/${EXPERIMENT}/${OPTIMIZER}/run-${SGE_TASK_ID}
echo `pwd`

export MODELPKL=%s
export PORTFILE=$TMPDIR/port.file

echo "Start Daemon at: `date`"
surrogate_daemonizer.py --pkl ${MODELPKL} --dir $TMPDIR/ --pid ${JOB_ID}_${SGE_TASK_ID} --start

echo "Sleep for 60s at: `date`"
sleep 60

echo "Check daemon at: `date`"
surrogate_daemonizer.py --pkl ${MODELPKL} --dir $TMPDIR/ --pid ${JOB_ID}_${SGE_TASK_ID} --status

ACLIB_ROOT=%s
ACLIB_VALIDATE_DIR=%s
TRAJ_FILE=%s
MYSQL_POOL=ACLIB_NIPS_surrogate

TAE="%s"

VALIDATE_PARAMS="--max-timestamp -1.0 --min-timestamp 0.0 --mult-factor 2.0 --validateOnlyLastIncumbent false "
PATH_PARAMS="--trajectoryFile ${TRAJ_FILE} --useScenarioOutDir true --outputDirectory ./validate-time --scenarioFile ./scenarios/*/${EXPERIMENT}/scenario.txt --algo-exec-dir ${ACLIB_VALIDATE_DIR}"

${ACLIB_ROOT}/configurators/smac-v2.10.02-master-773/smac-validate ${VALIDATE_PARAMS} ${PATH_PARAMS} ${TAE} --numRun ${SGE_TASK_ID} --num-validation-runs 1

echo "Job finished with exit code $? at: `date`"
"""


def modify_scenario(scenario_file):
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
                new_fl.write("algo = surrogate_communicator.py\n")
                algo = True
                algo_exec = line.strip().split("=")[1]
            elif not time and "wallclock-limit" in line:
                new_fl.write(line)
                limit = line.strip().split("=")[1]
                limit = int(float(limit.strip()))
                time = True
                new_fl.write("tunerTimeout = %d\n" % limit)
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

    if not time:
        logger.critical("Could not find 'wallclock-limit' in %s" % scenario_file)
        sys.exit(1)

    # Then overwrite
    shutil.move(scenario_file, scenario_file + ".original")
    shutil.move(scenario_file + ".new", scenario_file)

    return pcs_file, algo_exec, os.path.abspath(scenario_file)


def create_pbs_file(scenario_name, configurator, aclib_root, model, save,
                    sge_out, json):
    assert configurator in ("SMAC", "ParamILS", "ROAR", "SMAC --adaptive-capping false")
    print(sge_out, sge_out, scenario_name, model, configurator, json, aclib_root)
    pbs = pbs_template % (sge_out, sge_out, sync_virtualenv, scenario_name,
                          model, configurator, json, aclib_root)
    fh = open(save, "w")
    fh.write(pbs)
    fh.close()
    return


def create_validate_pbs_file(configurator, aclib_root, aclib_validate_root,
                             scenario_name, sge_out, working_dir,
                             algo_exec, pkl, save):
    traj_file = None
    if configurator == "ParamILS":
        traj_file = "./paramils_out/focused-*-traj_*.csv"
    elif configurator in ["SMAC", "ROAR", "SMAC --adaptive-capping false"]:
        traj_file = "./smac-output/aclib/traj-run-*.txt"
    else:
        logger.critical("%s is not a known configurator" % configurator)
        raise ValueError("%s is not a known configurator" % configurator)

    if pkl is None:
        tae = "--tae $TAE --mysqldbtae-pool ${MYSQL_POOL} " \
              "--mysqlTaeDefaultsFile ${HOME}/.aeatk/mysqldbtae.opt " \
              "--wait-for-persistent-run-completion false " \
              "--output-file-suffix worker --validate-by-wallclock-time false"
    else:
        tae = "--tae CLI --output-file-suffix local " \
              "--validate-by-wallclock-time false"
    pbs = pbs_validate_template % (sge_out, sge_out, sync_virtualenv,
                                   scenario_name, configurator, working_dir,
                                   pkl, aclib_root, aclib_validate_root,
                                   traj_file, algo_exec, tae)
    fh = open(save, "w")
    fh.write(pbs)
    fh.close()
    return


def create_validate_daemon_pbs_file(configurator, aclib_root,
                                    aclib_validate_root, scenario_name,
                                    sge_out, working_dir, pkl,
                                    save):
    traj_file = None
    if configurator == "ParamILS":
        traj_file = "./paramils_out/focused-*-traj_*.csv"
    elif configurator in ["SMAC", "ROAR", "SMAC --adaptive-capping false"]:
        traj_file = "./smac-output/aclib/traj-run-*.txt"
    else:
        logger.critical("%s is not a known configurator" % configurator)
        raise ValueError("%s is not a known configurator" % configurator)

    if pkl is None:
        tae = "--tae $TAE --mysqldbtae-pool ${MYSQL_POOL} " \
              "--mysqlTaeDefaultsFile ${HOME}/.aeatk/mysqldbtae.opt " \
              "--wait-for-persistent-run-completion false " \
              "--output-file-suffix worker --validate-by-wallclock-time false"
    else:
        tae = "--tae CLI --output-file-suffix local " \
              "--validate-by-wallclock-time false"
    pbs = pbs_daemon_template % (sge_out, sge_out, sync_virtualenv,
                                 scenario_name, configurator, working_dir, pkl,
                                 aclib_root, aclib_validate_root, traj_file,
                                 tae)
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
    parser.add_argument('--json', dest='json', required=True,
                        help="config json file")
    parser.add_argument('--save', dest='save', required=True,
                        help="Where to create Folder?")
    parser.add_argument('--aclib', dest='aclib_root', required=True,
                        help="Which aclib root to use?")
    parser.add_argument('--debug', dest="debug", default=False,
                        action="store_true", help="Show some more output")
    parser.add_argument('--aclibValidate', dest="aclib_root_validate",
                        default=None, help="Which aclib root to use for "
                                           "validation, if None use "
                                           "aclib_root + '_validate'")
    parser.add_argument('--sgeout', dest="sge_out", default="/home/eggenspk",
                        help="SGE output: sge_out + '/sgeout'")
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

    if not os.path.isfile(args.json):
        logger.error("%s does not exist" % args.json)
        sys.exit(1)
    else:
        args.json = os.path.abspath(args.json)

    if not os.path.isdir(args.aclib_root):
        logger.error("%s does not exist" % args.aclib_root)
        sys.exit(1)
    else:
        args.aclib_root = os.path.abspath(args.aclib_root)

    if args.aclib_root_validate is None:
        args.aclib_root_validate = args.aclib_root + "_validate"
        if not os.path.isdir(args.aclib_root_validate):
            logger.error("%s does not exist" % args.aclib_root_validate)
            sys.exit(1)
    else:
        args.aclib_root_validate = os.path.abspath(args.aclib_root_validate)

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
            scenario_file=os.path.join(new_scenario_dir, "scenario.txt"))

    # Create pbs File for ParamILS, SMAC
    for conf in ["SMAC", "ParamILS", "ROAR"]: #, "SMAC --adaptive-capping false"]:
        create_pbs_file(scenario_name=new_scenario_name, configurator=conf,
                        aclib_root=args.aclib_root, model=args.model,
                        save=os.path.join(args.save, new_scenario_name,
                                          ("run%s.pbs" % conf).replace(" ", "_")),
                        sge_out=args.sge_out, json=args.json)
        working_dir = os.path.abspath(os.path.join(args.save, new_scenario_name))
        create_validate_pbs_file(configurator=conf,
                                 aclib_root=args.aclib_root,
                                 aclib_validate_root=args.aclib_root_validate,
                                 scenario_name=new_scenario_name,
                                 algo_exec=algo_exec, sge_out=args.sge_out,
                                 working_dir=working_dir, pkl="NONE",
                                 save=os.path.join(args.save, new_scenario_name,
                                                   ("validateTrue%s.pbs" % conf)).replace(" ", "_"))
        create_validate_pbs_file(configurator=conf,
                                 aclib_root=args.aclib_root,
                                 aclib_validate_root=args.aclib_root_validate,
                                 scenario_name=new_scenario_name,
                                 algo_exec="surrogate_external_communicator.py",
                                 sge_out=args.sge_out,
                                 working_dir=working_dir, pkl=args.model,
                                 save=os.path.join(args.save, new_scenario_name,
                                                   ("validate%s.pbs" % conf)).replace(" ", "_"))
        create_validate_daemon_pbs_file(configurator=conf,
                                        aclib_root=args.aclib_root,
                                        aclib_validate_root=args.aclib_root_validate,
                                        scenario_name=new_scenario_name,
                                        sge_out=args.sge_out,
                                        working_dir=working_dir, pkl=args.model,
                                        save=os.path.join(args.save,
                                                          new_scenario_name,
                                                          ("validateDaemon%s.pbs" % conf)).replace(" ", "_"))

    # Create text for json File
    json_add = """
    "scenarios": {
        "%s": {
        "instances": "NOWHERE",
        "configurators": ["SMAC"],
        "scenario": "scenario.txt",
        "algorithm": "surrogate_communicator",
        "pcs": "%s"
        }
    },
    "algorithms": {
        "surrogate_communicator": {
        "download": "nowhere",
        "problem_type": ["sat"]
        }
    },
    "instances": {
        "NOTHING": {
            "download": "NOWHERE",
            "problem_type": "sat",
            "md5": "TODO"
        }
    }
    """

    logger.critical("You need to add the following to %s" % args.json)
    logger.critical("#"*80 + "\n" + json_add % (new_scenario_name, pcs_file) +
                    "\n" + "#"*80)
    logger.critical("And copy the scenario file from %s" %
                    os.path.join(new_scenario_dir, "scenario.txt"))

if __name__ == "__main__":
    main()