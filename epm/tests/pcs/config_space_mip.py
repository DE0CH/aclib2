"""
Created on May 15, 2015

@author: Andre Biedenkapp
"""

import logging
import os
import unittest

from epm.pcs.config_space import ConfigSpace


class Test(unittest.TestCase):

    def setUp(self):
        self.src_dir = os.path.dirname(os.path.dirname(__file__))
        logging.basicConfig(level=logging.DEBUG)

    def test_mip_cplex12(self):
        
        data_file = os.path.join(self.src_dir, "files", "txt", "cplex12-params-CPAIOR-space.txt")
        cs = ConfigSpace(data_file, normalize=False)
        
        def_config = cs.get_default_config_dict()
        print(def_config)
        def_vec = cs.convert_param_dict(def_config)
        def_config_back = cs.convert_param_vector(def_vec)
        
        self.assertEqual(len(cs.parameters), 76)
        
        for param, value in def_config.iteritems():
            assert value == def_config_back[param], "%s: %s vs %s" %(param, str(value), str(def_config_back[param]))

if __name__ == "__main__":
    unittest.main()