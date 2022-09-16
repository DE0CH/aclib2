import os
from subprocess import Popen
import subprocess
import logging
from multiprocessing.pool import ThreadPool

__author__ = "Marius Lindauer"
__version__ = "0.0.1"
__license__ = "BSD"

class LocalSystem(object):
    
    def __init__(self):
        '''
            Constructor
        '''
        
        self.template = ""
        self.logger = logging.getLogger("LocalSystem")
        
    def run(self, exp_dir: str, cmds:list, cores_per_job:int=1, job_cutoff:int=172800):
        '''
            runs the given command line calls (e.g., locally or by submitting jobs)
            
            Arguments
            ---------
            exp_dir: str
                execution directory of job
            cmds: list
                command line calls for AC experiments
            cores_per_job: int
                number of cores per job
            job_cutoff: int
                runtime cutoff per job
        '''
        
        #write cmds to disk
        cmd_fn = os.path.join(exp_dir, "cmds.txt")
        with open(cmd_fn, "w") as fp:
            for cmd in cmds:
                fp.write("'%s'\n" %(cmd))
        

        def send_cmd(cmd):
            self.logger.info(cmd)
            subprocess.run(cmd, shell=True)

        with ThreadPool(cores_per_job) as pool:
            pool.map(send_cmd, cmds)

