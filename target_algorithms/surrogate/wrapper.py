#!/usr/bin/env python
# encoding: utf-8

"""
lpg-wrapper -- Target algorithm wrapper for the version of LPG used in the
ICAPS'2013 paper
'Improved Features for Runtime Prediction of Domain-Independent Planners'

@author:     Chris Fawcett
@copyright:  2014 Chris Fawcett. All rights reserved.
@license:    BSD
@contact:    fawcettc@cs.ubc.ca

Example Call:
python <path to wrapper>/wrapper.py <instance>
<domain> <runtime cutoff> <runlength cutoff> <seed>
<--quality quality> <--model path to pickled model> <--idle_time idle time>
"""
import os
import random
import sys

from subprocess import Popen
from tempfile import NamedTemporaryFile
from genericWrapper4AC.generic_wrapper import AbstractWrapper


class SurrogateWrapper(AbstractWrapper):
    """
    Simple AC-wrapper for Surrogates
    """
    def __init__(self):
        self._tmp_dir = None
        self._ta_status = None
        self._ta_misc = None

        super(SurrogateWrapper, self).__init__()

        self.__script_dir = os.path.abspath(os.path.split(__file__)[0])
        self.script_dir = os.path.abspath(os.path.split(__file__)[0])

        # Extend parser with the following arguments. They need to be specified
        # in the corresponding scenario.txt file
        self.parser.add_argument("--quality", dest="quality", default="0",
                                 help="Does this scenario optimize quality? "
                                      "0 if False, 1 else")
        self.parser.add_argument('--pyrfr_wrapper', dest='pyrfr_wrapper',
                                 default=None,
                                 required=True, help='Path to Pyrfr Wrapper')
        self.parser.add_argument('--pyrfr_model', dest='pyrfr_model',
                                 default=None, required=True,
                                 help='Path to binary Random Forest (pyrfr)')
        self.parser.add_argument('--config_space', dest='config_space',
                                 default=None, required=True,
                                 help='Path to config space file')
        self.parser.add_argument('--inst_feat_dict', dest='inst_feat_dict',
                                 default=None, required=True,
                                 help='Path to instances-feature-dictionary in '
                                      'json format')
        self.parser.add_argument("--idle_time", default=600, type=int,
                                 help="Idle time until the server will be "
                                      "killed")
        self.parser.add_argument('--debug', type=str, default='False')

    def get_command_line_args(self, runargs, config):
        """
        Returns the command line call string to execute the target algorithm

        Parameters
        ----------
        runargs : dict
            A map of several optional arguments for the execution of
            the target algorithm.
                    {
                      "instance": <instance>,
                      "specifics" : <extra data associated with the instance>,
                      "cutoff" : <runtime cutoff>,
                      "runlength" : <runlength cutoff>,
                      "seed" : <seed>,
                      "idle_time": <idle time>
                    }
        config : dict
            A mapping from parameter name to parameter value

        Returns
        -------
            A command call list to execute the target algorithm.
        """

        self._tmp_dir = os.path.abspath(runargs["tmp"])

        pyrfr_wrapper = os.path.abspath(self.args.pyrfr_wrapper)
        pyrfr_model = os.path.abspath(self.args.pyrfr_model)
        config_space = os.path.abspath(self.args.config_space)
        inst_feat_dict = os.path.abspath(self.args.inst_feat_dict)

        if self.args.quality == "1":
            cmd = ["python", self.__script_dir + "/surrogate_communicator.py",
                   "QUALITY",
                   pyrfr_wrapper,
                   pyrfr_model,
                   config_space,
                   inst_feat_dict,
                   runargs["instance"],
                   "None",
                   str(runargs["cutoff"]),
                   str(runargs["runlength"]),
                   str(runargs["seed"]),
                   str(self.args.idle_time),
                   str(self.args.debug)]
        else:

            cmd = ["python", self.__script_dir + "/surrogate_communicator.py",
                   pyrfr_wrapper,
                   pyrfr_model,
                   config_space,
                   inst_feat_dict,
                   runargs["instance"],
                   runargs["specifics"] if runargs.get('specifics', '') not in [None, ''] else "None",
                   str(runargs["cutoff"]),
                   str(runargs["runlength"]),
                   str(runargs["seed"]),
                   str(self.args.idle_time),
                   str(self.args.debug)]

        for key, value in config.items():
            cmd.extend([key, value])

        return " ".join(cmd)

    def process_results(self, filepointer, out_args):
        """
        Parse a results file to extract the run's status (SUCCESS/CRASHED/etc)
        and other optional results.

        Parameters
        ----------
        filepointer :
            a pointer to the file containing the solver execution
            standard out.
        out_args :
            a map with {"exit_code" : exit code of target algorithm}
        Returns
        -------
        dict
            A map containing the standard AClib run results. The current
            standard result map as of AClib 2.06 is:
            {
                "status" : <"SAT"/"UNSAT"/"TIMEOUT"/"CRASHED"/"ABORT">,
                "runtime" : <runtime of target algrithm>,
                "quality" : <a domain specific measure of the quality of the
                            solution [optional]>,
                "misc" : <a (comma-less) string that will be associated with
                         the run [optional]>
            }
            ATTENTION: The return values will overwrite the measured results of
            the runsolver (if runsolver was used).
        """
        result_map = {}

        for line in filepointer:
            line = str(line)
            if "Result for ParamILS" in line:
                line = line.split(",")
                result_map['status'] = line[0].split(":")[1].strip()
                result_map['runtime'] = float(line[1])
                result_map['quality'] = float(line[2])

                result_map['misc'] = 'Seed:{}-Duration:{}-Address:{}-Port:{}' \
                    .format(line[3], line[4], line[5], line[6])
        self.logger.info('Return Result map: {}'.format(result_map))
        return result_map

    def call_target(self, target_cmd):
        """
        Extends the target algorithm command line call with the runsolver
        and executes it

        Parameters
        ----------
        target_cmd : list[str)
            list of target cmd (from getCommandLineArgs)
        """
        random_id = random.randint(0, 1000000)
        self._watcher_file = NamedTemporaryFile(
                suffix=".log", prefix="watcher-%d-" % random_id,
                dir=self._tmp_dir, delete=False)
        self._solver_file = NamedTemporaryFile(
                suffix=".log", prefix="solver-%d-" % random_id,
                dir=self._tmp_dir, delete=False)

        runsolver_cmd = []

        try:
            if self.data.runsolver != "None":
                runsolver_cmd = [self.data.runsolver,
                                 "-M", self.data.mem_limit,
                                 "-C", self.data.cutoff,
                                 "-w", "\"%s\"" % self._watcher_file.name,
                                 "-o",  "\"%s\"" % self._solver_file.name]
        except AttributeError as e:
            self.logger.error(
                    'Some Attribute Error in the Wrapper occurred:', e
                    )

        runsolver_cmd = " ".join(map(str, runsolver_cmd)) + " " + target_cmd

        self.logger.info('Run solver is {}'.format(
                ('active' if self.data.runsolver != 'None' else 'deactivated'))
                )
        self.logger.debug("Calling command-line args {}:".format(runsolver_cmd))

        try:
            if self.data.runsolver != "None":
                io = Popen(runsolver_cmd, shell=True, preexec_fn=os.setpgrp,
                           universal_newlines=True)
            else:
                io = Popen(runsolver_cmd, stdout=self._solver_file, shell=True,
                           preexec_fn=os.setpgrp, universal_newlines=True)
            self._subprocesses.append(io)
            io.wait()
            self._subprocesses.remove(io)
            if io.stdout:
                io.stdout.flush()
        except OSError as e:
            self._ta_status = "ABORT"
            self._ta_misc = "execution failed: %s" \
                            % (" ".join(map(str, runsolver_cmd)))
            self._exit_code = 1
            self.logger.error('ABORT due to OSError {}'.format(e))
            sys.exit(1)

        self._solver_file.seek(0)


if __name__ == "__main__":
    wrapper = SurrogateWrapper()
    wrapper.main()
