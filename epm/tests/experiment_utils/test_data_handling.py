"""
Created on Mar 27, 2015

@author: Katharina Eggensperger
"""
import copy
import unittest

import numpy
from epm.experiment_utils.data_handling import partition_data


class dataHandlingTest(unittest.TestCase):

    def test_partition_data(self):
        seed = 1

        # Simple example
        data_matrix = numpy.array([[1, 2, 5.3], [1, 2, 5.3], [1, 2, 5.2], [1, 2, 5.2]])
        inst_list = ["inst1", "inst2", "inst1", "inst2"]

        rs = numpy.random.RandomState(seed)
        indices = partition_data(data_matrix=data_matrix, inst_list=inst_list,
                                 rs=rs)
        self.assertListEqual(indices, [[[2], [3]], [[0], [1]]])
        # Check for consistent results
        rs = numpy.random.RandomState(seed)
        indices = partition_data(data_matrix=data_matrix, inst_list=inst_list,
                                 rs=rs)
        self.assertListEqual(indices, [[[2], [3]], [[0], [1]]])

        # unbalanced example
        data_matrix = numpy.array([[1, 2, 5.3], [1, 2, 5.3], [1, 2, 5.2], [1, 2, 5.2], [1, 2, 5.2]])
        inst_list = ["inst1", "inst2", "inst1", "inst2", "inst3"]
        indices = partition_data(data_matrix=data_matrix, inst_list=inst_list,
                                 rs=numpy.random.RandomState(seed))
        self.assertListEqual(indices, [[[3, 4], [2]], [[1], [0]]])
        # Check for consistent results
        indices = partition_data(data_matrix=data_matrix, inst_list=inst_list,
                                 rs=numpy.random.RandomState(seed))
        self.assertListEqual(indices, [[[3, 4], [2]], [[1], [0]]])

        # example with two similar inputs
        data_matrix = numpy.array([[1, 2, 5.3], [1, 2, 5.3], [1, 2, 5.2], [1, 2, 5.2], [1, 2, 5.2]])
        inst_list = ["inst1", "inst2", "inst1", "inst2", "inst2"]
        seed = 1
        indices = partition_data(data_matrix=data_matrix, inst_list=inst_list,
                                 rs=numpy.random.RandomState(seed))
        self.assertListEqual(indices, [[[2], [3, 4]], [[0], [1]]])

        # example with only zeros and ones
        data_matrix = numpy.array([[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0]])
        inst_list = ["inst1", "inst2", "inst1", "inst2", "inst2"]
        seed = 1
        self.assertRaisesRegexp(ValueError, "retry with different seed",
                                partition_data, data_matrix=data_matrix,
                                inst_list=inst_list,
                                rs=numpy.random.RandomState(seed), debug=True)

        # This is not what we want
        indices = partition_data(data_matrix=data_matrix, inst_list=inst_list,
                                 rs=numpy.random.RandomState(seed+2))
        self.assertListEqual(indices, [[[2], [1]], [[0], [3, 4]]])

        # Example with large config
        rs = numpy.random.RandomState(seed=seed)
        config1 = rs.rand(1, 1000)
        config2 = rs.rand(1, 1000)
        config3 = rs.rand(1, 1000)
        data_matrix = numpy.array([config1, config2, config3, config1])
        inst_list = ["inst1", "inst2", "inst1", "inst2", "inst2"]
        seed = 1
        # We expect a similar error
        self.assertRaisesRegexp(ValueError, "retry with different seed",
                                partition_data, data_matrix=data_matrix,
                                inst_list=inst_list,
                                rs=numpy.random.RandomState(seed), debug=True)
        indices = partition_data(data_matrix=data_matrix, inst_list=inst_list,
                                 rs=numpy.random.RandomState(seed+6))
        self.assertListEqual(indices, [[[3], [0]], [[1], [2]]])

        config2 = copy.deepcopy(config1)
        config2[0, 500] += 0.00000000000001
        config3 = copy.deepcopy(config1)
        config3[0, 501] += 0.00000000000001
        data_matrix = numpy.array([config1, config2, config3, config1])
        # We expect a similar error
        self.assertRaisesRegexp(ValueError, "retry with different seed",
                                partition_data, data_matrix=data_matrix,
                                inst_list=inst_list,
                                rs=numpy.random.RandomState(seed), debug=True)
        indices = partition_data(data_matrix=data_matrix, inst_list=inst_list,
                                 rs=numpy.random.RandomState(seed+3))
        self.assertListEqual(indices, [[[0], [3]], [[2], [1]]])




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testILS']
    unittest.main()
