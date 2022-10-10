"""
Created on Mar 25, 2015

@author: manju
"""

import logging
import numpy

class FeatureParser(object):
    """
        reads instance feature files
    """


    def __init__(self):
        """
        Constructor
        """
    
    def read_features(self, feature_file, n_feats=-1):
        """
            read feature file.
            Warning: Missing features are probably encoded as -512.
            Args:
                feature_file: csv file with format: <instance name>, <feat_1>, <feat_2>,...
                n_feats: number of features (if set to < 0, use number found in data)
        """

        #TODO: impute missing values
        logging.debug("Read %s" % feature_file)

        inst_feat_dict = {} # instance -> features 
        n_feats_used = None

        with open(feature_file) as fp:
            header = fp.readline()
            if n_feats < 0 :
                n_feats_used = len(header.strip("\n").split(",")) - 1
            else:
                n_feats_used = min(len(header.strip("\n").split(",")) - 1, n_feats)
                
            for line in fp:
                parts = line.strip("\n").split(",")
                instance = parts[0]
#                features = map(float,parts[1:n_feats_used+1])
                features = list(map(numpy.float32, parts[1:n_feats_used+1]))
                
                inst_feat_dict[instance] = features   
        logging.debug("Used features: %d" % n_feats_used)
                
        return inst_feat_dict, n_feats_used
