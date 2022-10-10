"""
Created on May 19, 2015

@author: Andre Biedenkapp
"""

import logging
import os
import unittest

from epm.pcs.config_space import ConfigSpace
from epm.reader.matrix_parser import MatrixParser


class Test(unittest.TestCase):

    def setUp(self):
        self.src_dir = os.path.dirname(os.path.dirname(
                                       os.path.abspath(__file__)))
        logging.basicConfig(level=logging.INFO)
        self.mp = MatrixParser()

    @unittest.skip("Don't do matlab data tests")
    def test_mip_cplex12_perf_matrix(self):
        ### This can be deleted as soon as someone has checked it
        # We now return a TIMEOUT_list instead of a succ_list when reading data
        print("TODO check whether timeout_matrix is built correctly")
        if "CODESHIP" in os.environ:
            return
        ####
        data_file = os.path.join(self.src_dir, "files", "txt", "10samples-CPLEX-BIGMIX-100inst-results.txt")
        perf_list, suc_list, cen_list, inst_list, best_perf, ic = self.mp.read_performance_matrix(data_file)
        
        self.assertEqual(ic[0], 100)
        self.assertEqual(ic[1], 10)
        
        self.assertEqual(len(best_perf.keys()), 100)
        
        self.assertEqual(best_perf['/ubc/cs/project/arrow/projects/MIP_data/coral-June10-2010/neos-953928.mps'],
                         111.37)
        self.assertEqual(best_perf['/ubc/cs/project/arrow/projects/MIP_data/coral-June10-2010/neos-1441553.mps'],
                         2.64)
        
        self.assertEqual(len(perf_list), 100*10)
        self.assertEqual(len(inst_list), 100*10)
        self.assertEqual(len(suc_list), 100*10)
        self.assertEqual(len(cen_list), 100*10)
        
        self.assertEqual(perf_list[0], 3000)
        self.assertEqual(suc_list[0], False)
        self.assertEqual(cen_list[0], False)
        self.assertEqual(inst_list[0], '/ubc/cs/project/arrow/projects/MIP_data/coral-June10-2010/neos-953928.mps')
        
        self.assertEqual(perf_list[3], 162.98)
        self.assertEqual(suc_list[3], True)
        self.assertEqual(cen_list[3], False)
        self.assertEqual(inst_list[3], '/ubc/cs/project/arrow/projects/MIP_data/coral-June10-2010/neos-826812.mps')

    @unittest.skip("Don't do matlab data tests")
    def test_censoring(self):
        ### This can be deleted as soon as someone has checked it
        # We now return a TIMEOUT_list instead of a succ_list when reading data
        print("TODO check whether timeout_matrix is built correctly")
        if "CODESHIP" in os.environ:
            return
        ####
        data_file = os.path.join(self.src_dir, "files", "txt", "10samples-CPLEX-BIGMIX-100inst-results.txt")
        pperf_list, ssuc_list, ccen_list, iinst_list, bbest_perf, ic = self.mp.read_performance_matrix(data_file)
        
        for i in bbest_perf.keys():
            self.assertTrue(bbest_perf[i] < 3000)
            logging.info('Best for ...%s is %.3f' %(i[-10:], bbest_perf[i]))
        #######################################################################
        #######################################################################
        perf_list, cen_list = self.mp.censor_data(pperf_list, ccen_list, iinst_list, bbest_perf, 300, 'fixed')
        
        self.assertEqual(len(perf_list), 100*10)
        self.assertEqual(len(cen_list), 100*10)
        
        self.assertEqual(perf_list[0], 300)
        self.assertEqual(cen_list[0], True)
        
        self.assertEqual(perf_list[3], 162.98)
        self.assertEqual(cen_list[3], False)
        
        for i in range(len(cen_list)):
            if cen_list[i]:
                self.assertEqual(perf_list[i], 300)
            else:
                self.assertTrue(perf_list[i] != 300)
        
        #######################################################################
        #######################################################################
        perf_list, cen_list = self.mp.censor_data(pperf_list, ccen_list, iinst_list, bbest_perf, 300, 'random')
        
#        self.assertEqual(len(perf_list), 100*10)
#        self.assertEqual(len(cen_list), 100*10)
        
#        self.assertTrue((perf_list[0] > 0) and (perf_list[0] < 300))
#        self.assertEqual(cen_list[0], True)
        count = 0
        for i in cen_list:
            if i:
                count += 1
        logging.info('Num censored data = %d ' %count)
        #######################################################################
        #######################################################################
        del perf_list, cen_list
        print('capslack')
        perf_list, cen_list = self.mp.censor_data(pperf_list, ccen_list, iinst_list, bbest_perf, 300, 'capslack')
        
        self.assertEqual(len(perf_list), 100*10)
        self.assertEqual(len(cen_list), 100*10)
        
#        self.assertTrue(perf_list[0] <= 300)
#        self.assertEqual(cen_list[0], True)
        
#        self.assertTrue(perf_list[3] <= 162.98 or perf_list[3] == 300)
#        self.assertEqual(cen_list[3], True)
        
        print(perf_list[0:10])
        count = 0
        for i in cen_list:
            if i:
                count += 1
        print(count)

    @unittest.skip("Don't do matlab data tests")
    def test_mip_cplex12_config_matrix(self):
        ### This can be deleted as soon as someone has checked it
        # We now return a TIMEOUT_list instead of a succ_list when reading data
        print("TODO check whether timeout_matrix is built correctly")
        if "CODESHIP" in os.environ:
            return
        ####
        config_space_file = os.path.join(self.src_dir, "files", "txt", "cplex12-params-CPAIOR-space.txt")
        cs = ConfigSpace(config_space_file, normalize=False)
        
        random_config_file = os.path.join(self.src_dir, "files", "txt", "1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt")
        config_list=self.mp.read_config(random_config_file, 1510)
        
        self.assertEqual(len(config_list), 1000*1510)
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
