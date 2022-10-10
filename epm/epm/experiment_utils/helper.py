"""
Created on Aug 25, 2015

@author: Katharina Eggensperger
"""

from epm.preprocessing.pre_feature import PreprocessingFeatures


def get_feature_normalizer(train_lists, inst_feat_dict):
    # returns feature normalizer for instances in train list
    tmp_inst_feat_dict = dict()
    for l in train_lists:
        if l is None:
            continue
        for i in l:
            tmp_inst_feat_dict[i] = inst_feat_dict[i]
    preproc = PreprocessingFeatures(tmp_inst_feat_dict)
    return preproc
