#!/usr/bin/env python2.7
# encoding: utf-8

'''
cplexWrapper -- AClib target algorithm warpper for MIP solver CPLEX (version 12.6)

@author:     Marius Lindauer, Chris Fawcett, Alex Fréchette, Frank Hutter
@copyright:  2014 AClib. All rights reserved.
@license:    BSD
@contact:    lindauer@informatik.uni-freiburg.de, fawcettc@cs.ubc.ca, afrechet@cs.ubc.ca, fh@informatik.uni-freiburg.de

example call (in aclib folder structure):
python src/generic_wrapper/cplexWrapper.py --runsolver-path /data/aad/aclib/target_algorithms/runsolver/runsolver --mem-limit 1024 /data/aad/aclib/instances/mip/data/COR-LAT/COR-LAT/cor-lat-2f+r-u-10-10-10-5-100-3.013.b58.000000.prune2.lp ee 20 -1 12345 -simplex_perturbation_switch YES -perturbation_constant 0.0000001
'''

import sys
import re
import os
import math

from genericWrapper4AC.generic_wrapper import AbstractWrapper

class MipWrapper(AbstractWrapper):
    '''
        Simple wrapper for a MIP solver (CPLEX)
    '''
    
    def __init__(self):
        AbstractWrapper.__init__(self)
        
        self.parser.add_argument("--obj-file", dest="obj_file", default=None, help="file with optimal objectives for each instance ")

    
    def get_command_line_args(self, runargs, config):
        '''
        Returns the command line call string to execute the target algorithm (here: Spear).
        Args:
            runargs: a map of several optional arguments for the execution of the target algorithm.
                    {
                      "instance": <instance>,
                      "specifics" : <extra data associated with the instance>,
                      "cutoff" : <runtime cutoff>,
                      "runlength" : <runlength cutoff>,
                      "seed" : <seed>
                    }
            config: a mapping from parameter name to parameter value
        Returns:
            A command call list to execute the target algorithm.
        '''
        binary = "target_algorithms/mip/cplex12.6/cplex"
        
        params = []
        if config["-simplex_perturbation_switch"] == "yes":
            simplex_perturbation_value = config["-simplex_perturbation_switch"] + " " + config["-perturbation_constant"]
        else:
             simplex_perturbation_value = config["-simplex_perturbation_switch"] + " 0.00000001"
        params.append("set simplex perturbationlimit %s " % (simplex_perturbation_value) )
        try:
            del config["-simplex_perturbation_switch"]
            del config["-perturbation_constant"]
        except KeyError:
            pass
        
        for name, value in config.items():
            params.append("set %s %s" %(name.replace("_", " ")[1:],  value))

         # "set logfile *" disables the log file; important to not fill the disk
        metaparams = [
            "set logfile *",
            "read %s" % runargs["instance"],
            "set clocktype 1",
            "set threads 1",
            "set timelimit %d" % runargs["cutoff"],
            "set mip limits treememory 900",
            "set workdir .",
            "set mip tolerances mipgap 0"]
        if runargs['seed'] != -1:
            metaparams.append("set randomseed %d" % runargs['seed'])

        if "_obj_max" in runargs["instance"]:
            metaparams.append("change sense obj max")
        
        
        metaparams.extend( params )
        metaparams.append( "display settings all" )
        metaparams.append( "opt" )
        metaparams.append( "quit" )
               
        return binary + " -c \"" + "\" \"".join(metaparams) +"\""
    
    def process_results(self, filepointer, out_args):
        '''
        Parse a results file to extract the run's status (SUCCESS/CRASHED/etc) and other optional results.
    
        Args:
            filepointer: a pointer to the file containing the solver execution standard out.
            out_args : a map with {"exit_code" : exit code of target algorithm}
        Returns:
            A map containing the standard AClib run results. The current standard result map as of AClib 2.06 is:
            {
                "status" : <"SAT"/"UNSAT"/"TIMEOUT"/"CRASHED"/"ABORT">,
                "runtime" : <runtime of target algrithm>,
                "quality" : <a domain specific measure of the quality of the solution [optional]>,
                "misc" : <a (comma-less) string that will be associated with the run [optional]>
            }
            ATTENTION: The return values will overwrite the measured results of the runsolver (if runsolver was used). 
        '''
        self.logger.info("reading solver results from %s" % (filepointer.name))
        filepointer.seek(0)
        #data = filepointer.read()
        resultMap = {}

        optimal_objective_file = self.args.obj_file  
        opt_perf = 0
        with open(optimal_objective_file, "r") as fp:
            for line in fp:
                if out_args["instance"] in line: # hack as long as opt_obj_file is not fixed
                    opt_perf = float(line.split(" ")[1])
                    
        
        #numeric_const_pattern = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
        numeric_const_pattern = r"[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?"
        
        gap = None
        internal_measured_runtime = None
        iterations = None
        measured_runlength = None
        obj = None
        solved = None
        
        for line in filepointer:
            line = line.decode("utf8")
            if re.match("gap = (%s), (%s)" %(numeric_const_pattern, numeric_const_pattern),line):
                 m = re.match("gap = %s, (%s)" %(numeric_const_pattern, numeric_const_pattern),line)
                 gap = float(m.group(0))
            elif re.match("^Solution time\s*=\s*(%s)\s*sec\.\s*Iterations\s*=\s*(\d+)\s*Nodes\s*=\s*(\d+)$" %(numeric_const_pattern), line):
                 m = re.match("Solution time\s*=\s*(%s)\s*sec\.\s*Iterations\s*=\s*(\d+)\s*Nodes\s*=\s*(\d+)" %(numeric_const_pattern), line)
                 internal_measured_runtime = float(m.group(1))
                 iterations = int(m.group(2))
                 measured_runlength = int(m.group(3))
            elif re.match("MIP\s*-\s*Integer optimal solution:\s*Objective\s*=\s*(%s)" %(numeric_const_pattern), line):
                m = re.match("MIP\s*-\s*Integer optimal solution:\s*Objective\s*=\s*(%s)" %(numeric_const_pattern), line)
                gap = 0
                obj = float(m.group(1))
                solved = "SAT"
            elif re.match("MIP\s*-\s*Integer optimal,\s*tolerance\s*\(%s\/%s\):\s*Objective\s*=\s*(%s)" %(numeric_const_pattern, numeric_const_pattern, numeric_const_pattern), line):
                 m = re.match("MIP\s*-\s*Integer optimal,\s*tolerance\s*\(%s\/%s\):\s*Objective\s*=\s*(%s)" %(numeric_const_pattern, numeric_const_pattern, numeric_const_pattern), line)
                 obj = float(m.group(1))
                 solved = "SAT"
            elif re.match("Optimal:\s*Objective =\s*%s" %(numeric_const_pattern), line):
                solved = "SAT"
            elif re.match("No problem exists.", line): # instance could not be read
                solved = "ABORT"
            elif re.match("MIP\s*-\s*Time limit exceeded, integer feasible:\s*Objective\s*=\s*(%s)" %(numeric_const_pattern), line):
                m = re.match("MIP\s*-\s*Time limit exceeded, integer feasible:\s*Objective\s*=\s*(%s)" %(numeric_const_pattern), line)
                obj = float(m.group(1))
                solved = "TIMEOUT"
            elif re.match("MIP\s*-\s*Error termination, integer feasible:\s*Objective\s*=\s*(%s)" %(numeric_const_pattern), line):
                m = re.match("MIP\s*-\s*Error termination, integer feasible:\s*Objective\s*=\s*(%s)" %(numeric_const_pattern), line)
                obj = float(m.group(1))
                solved = "TIMEOUT"
            elif "MIP - Error termination, no integer solution." in line:
                solved = "TIMEOUT"
            elif "MIP - Time limit exceeded, no integer solution." in line:
                solved = "TIMEOUT"
            elif "CPLEX Error  1001: Out of memory." in line:
                solved = "TIMEOUT"
            elif "MIP - Memory limit exceeded" in line:
                solved = "TIMEOUT"

        #=======================================================================
        # print("Gap: %f" %(gap))
        # print("Measure time: %f" %(internal_measured_runtime))
        # print("Iterations: %d" %(iterations))
        # print("Measured runlength: %d" %(measured_runlength))
        # print("Obj: %f" %(obj))
        #=======================================================================

        if solved == "SAT":
            # check correctness
            slack_in_my_assertions = 1.0001 # small multiplicative slack in assertions
            maxi = max(math.fabs(obj), math.fabs(opt_perf)) 
            if math.fabs(obj - opt_perf)/maxi > slack_in_my_assertions and math.fabs(obj - opt_perf) > 1e-8:
                solved = "CRASHED" # "WRONG ANSWER
                self.logger.warning("CPLEX claims to have solved the instance, but its result %f differs from the actual one %f by more than a relative error of 0.01%." %(obj, opt_perf))
                resultMap["misc"] = "WRONG ANSWER: Gap too large"
            else:
                resultMap["misc"] = "Solution verified"
                
        if solved is not None:
            resultMap["status"] = solved
        if obj is not None:
            resultMap["quality"] = obj

        return resultMap

if __name__ == "__main__":
    wrapper = MipWrapper()
    wrapper.main()    
