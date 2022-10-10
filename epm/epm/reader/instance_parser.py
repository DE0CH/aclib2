"""
Created on Jun 23, 2015

@author: Katharina Eggensperger
"""

import logging
import os


class InstanceParser(object):
    """
        reads file listing instances
    """

    def __init__(self):
        """
        Constructor
        """

    def read_instances(self, instance_file):
        """
            read file listing instances
        :param instance_file: file with format: <instance name>\n<instance name>
        :return: list of instance strings
        """
        if not os.path.isfile(instance_file):
            raise ValueError("%s is not a file" % instance_file)

        logging.debug("Read %s" % instance_file)

        instance_list = list()
        fh = open(instance_file, 'r')
        for inst in fh.readlines():
            tmp = inst.strip().split(" ")

            # in case the solubility is given
            tmp = tmp[0]
            instance_list.append(tmp)
        fh.close()
        return instance_list