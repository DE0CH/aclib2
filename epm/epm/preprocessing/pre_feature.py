"""
Created on Mar 25, 2015

@author: manju
"""

import numpy
from sklearn.preprocessing import MinMaxScaler

class PreprocessingFeatures(object):
    """
       preproces features
    """

    def __init__(self, inst_feats_dict):
        """
        Constructor
        """
        feats = numpy.array([v for v in inst_feats_dict.values()])
        self.transformer = MinMaxScaler()
        self.transformer.fit(feats)

    def normalization(self, inst_feats_dict):
        """
            normalize features
        """
        
        instances = inst_feats_dict.keys()
        
        feats = numpy.array(list(inst_feats_dict[inst] for inst in instances))
        
        feats = self.transformer.transform(feats)
        
        new_inst_feats_dict = dict((inst, feats[idx]) for idx, inst in enumerate(instances))
        
        return new_inst_feats_dict