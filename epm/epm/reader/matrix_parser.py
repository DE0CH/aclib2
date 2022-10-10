'''
Created on May 19, 2015

@author: Andre Biedenkapp
'''

import logging
import numpy
import random as rand

class MatrixParser(object):
    '''
    reads collected performance data of target algorithm
    '''

    def __init__(self):
        self.logger=logging.getLogger('MPar')
        '''
        Constructor
        '''
        
    def read_performance_matrix(self, file_, matrix_captime = 300):
        '''
        Reads in the runtime-matrix file and generates a matrix from it.
        
        returns the runtime-matrix, a timeout_matrix, an instance_list
        
        A row in the runtime-matrix contains the n resulting runtimes for n
        configurations on a given instance.
        timeout_matrix is a boolean matrix of the same size as the runtime-matrix.
        Each entry is set to True, since we treat every runtime as uncensored.
        The instance_list contains the names of each instance.
        '''

        ### This can be deleted as soon as someone has checked it
        # We now return a TIMEOUT_list instead of a succ_list when reading data
        self.logger.critical("TODO check whether timeout_matrix is built "
                             "correctly")
        import os
        if "CODESHIP" not in os.environ:
            raise NotImplementedError("TODO check whether timeout_matrix is "
                                      "built correctly")
        ####

        perf_matrix = None
        timeout_matrix = list()
        censored_matrix = list()
        instance_list = list()
        with open(file_, 'rb') as perf_matrix_file:
            for line in perf_matrix_file:
                # remove whitespace, tabs and newlines
                line = line.strip(' \t\n\r')

                # get the individual entries
                parts = line.split(',')
                entries = parts[1:]
                instance = parts[0].strip(' ')
                timeout_list = []
                cen_list = []
                for e in xrange(len(entries)):
                    entries[e] = float(entries[e])

                    # Handle too small values
                    if entries[e] < 0.005:
                        entries[e] = 0.005

                    # Determine whether run was successful or not
                    if entries[e] > matrix_captime-0.01:
                        # This value is larger than captime, so it is a timeout
                        timeout_list.append(True)
                        #cen_list.append(True)
                    else:
                        timeout_list.append(False)
                        #cen_list.append(False)
                    # Matlab data never has censored runtimes
                    cen_list.append(False)
                timeout_list = numpy.array(timeout_list).reshape((len(timeout_list), 1))
                cen_list = numpy.array(cen_list).reshape((len(cen_list), 1))
                entries = numpy.array(entries).reshape((len(entries), 1))

                if perf_matrix is None:
                    # Do this only at the beginning
                    perf_matrix = entries
                    timeout_matrix = timeout_list
                    censored_matrix = cen_list
                else:
                    perf_matrix = numpy.hstack((perf_matrix, entries))
                    timeout_matrix = numpy.hstack((timeout_matrix, timeout_list))
                    censored_matrix = numpy.hstack((censored_matrix, cen_list))
                instance_list.append(instance)
        
        C = len(perf_matrix)        # number of Configurations
        I = len(perf_matrix[0])     # number of Instances
        
        perf_list = list()
        timeout_list = list()
        cen_list = list()
        best_perf = {}
        
        inst_list = instance_list

        # Build list with best performance per instance
        for i in range(len(inst_list)):
            best_perf[inst_list[i]] = float('inf')
        
        for i in range(len(perf_matrix)):
            for j in range(len(perf_matrix[0])):
                best_perf[inst_list[j]] = min(best_perf[inst_list[j]], perf_matrix[i][j])
                perf_list.append(perf_matrix[i][j])
                timeout_list.append(timeout_matrix[i][j])
                cen_list.append(censored_matrix[i][j])

        inst_list *= len(perf_matrix)
        del timeout_matrix, censored_matrix, perf_matrix, instance_list

        return perf_list, timeout_list, cen_list, inst_list, best_perf, (I, C)

    def censor_data(self, perf_list, cens_list, inst_list, best_perf, cap_time = 1, cap_type = 'fixed'):
        cap_time = float(cap_time)
        if cap_type == 'fixed': # commented in the matlab code with 'to visualize predictions'
            for i in range(len(perf_list)):
                if perf_list[i] > cap_time:
                    cens_list[i] = True
                    perf_list[i] = cap_time
        elif cap_type == 'random':
            count = 0
            for i in range(len(perf_list)):
                new_cap_time = 10.0**(-3 + (rand.random() * numpy.log10(cap_time) + 3))
                if perf_list[i] > new_cap_time:
                    count += 1
                    cens_list[i] = True
                    perf_list[i] = new_cap_time
            self.logger.info('Num censored data = %d' %count)
        elif cap_type == 'capslack':
            count = 0
            for i in range(len(perf_list)):
                if perf_list[i] > best_perf[inst_list[i]]*cap_time:
                    count += 1
                    perf_list[i] = min(best_perf[inst_list[i]]*cap_time, 300) # 300 taken from Franks original code/paper.
                    cens_list[i] = True
            self.logger.info('Num censored data = %d' %count)
            
        del best_perf
        
        return perf_list, cens_list
        
    def read_config(self, file_, num_instances, whole_traj_file=0):
        '''
        Reads in the random config file and generates a parameter_index_matrix from it.
        
        The N parameter settings are in the .traj format (without the first 5 entries
        of .traj, unless whole_traj_file is set.)
        
        returns a list of dictionaries. Every dictionary holds a configuration.
        
        file_ the file name of the random_config_file
        whole_traj_file flag if the first 5 entries are in the .traj format
        '''
        
        if whole_traj_file: # if this is set, read the whole file
            start_idx = 0
        else:
            start_idx = 5   # if not we ignore the dummy values
        
        ###############################################################################
        # Read data
        ###############################################################################
        data=[]
        with open(file_, 'rb') as random_config_file:
            for line in random_config_file:
                line=line.strip(' \t\n\r')              # remove whitespace, tabs and newlines
                entries = line.split(',')[start_idx:]   # get the individual entries
                for e in range(len(entries)):
                    entries[e] = entries[e].strip(' ')  # remove any remaining whitespace
                data.append(entries)
        
        if whole_traj_file:
            for i in range(len(data)):
                data[i]=data[i][5:len(data[i])]
        
        config_list=[]
        
        for i in range(1,len(data)):
            conf_dict = {}
            for j in range(len(data[i])):
                conf_dict[data[0][j]]=data[i][j]
            for k in range(num_instances):
                config_list.append(conf_dict)
        
        return config_list
