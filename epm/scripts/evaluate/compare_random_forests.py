#!/usr/local/bin/python2.7
# encoding: utf-8
"""
EPM -- comparing sklearn random forest with stephans

@author:     Andre Biedenkapp

@copyright:  2015 AAD Group Freiburg. All rights reserved.

@license:    GPLv2

@contact:    {biedenka}@informatik.uni-freiburg.de
"""

# PAR10: penalized average runtime, counting timeouts at tmax as 10 ·tmax

import argparse

try:
    import cPickle as pickle
except ImportError:
    import pickle

import pickle
import logging
import os
import time
import numpy

from sklearn.metrics import mean_squared_error

from epm.experiment_utils.data_handling import loadMatlabData, \
    separate_data_with_bools, partition_data, build_data
from epm.experiment_utils.helper import get_feature_normalizer
from epm.models.model_dict import model_dict as model_dict
from epm.models.external_rfr import ExternalRFR
from epm.pcs.config_space import ConfigSpace

RNG = 0
numpy.random.seed(RNG)

def preprocess_data(train_idx, test_idx, encoded_matrix, inst_list, perf_list, cen_list, inst_feat_dict, n_feats_used):
    # Split data
    train_cen_encoded_matrix, train_cen_inst_list, train_cen_y, train_uncen_encoded_matrix, train_uncen_inst_list, train_uncen_y = \
        separate_data_with_bools(data_matrix=encoded_matrix[train_idx, :],
                                 inst_list=[inst_list[i] for i in train_idx],
                                 perf_list=[perf_list[i] for i in train_idx],
                                 succ_list=[cen_list[i] for i in train_idx])
    test_cen_encoded_matrix, test_cen_inst_list, test_cen_y, test_uncen_encoded_matrix, test_uncen_inst_list, test_uncen_y = \
        separate_data_with_bools(data_matrix=encoded_matrix[test_idx, :],
                                 inst_list=[inst_list[i] for i in test_idx],
                                 perf_list=[perf_list[i] for i in test_idx],
                                 succ_list=[cen_list[i] for i in test_idx])

    # Build some datasets
    feature_normalizer = get_feature_normalizer(train_lists=(train_cen_inst_list, train_uncen_inst_list), inst_feat_dict=inst_feat_dict)
    logging.info("# Got normalizer")
    inst_feat_dict = feature_normalizer.normalization(inst_feat_dict)
    logging.info("# Features are normalized")
    train_uncen_X = build_data(data_matrix=train_uncen_encoded_matrix,
                               inst_list=train_uncen_inst_list,
                               inst_feat_dict=inst_feat_dict,
                               n_feats=n_feats_used)
    test_uncen_X = build_data(data_matrix=test_uncen_encoded_matrix,
                              inst_list=test_uncen_inst_list,
                              inst_feat_dict=inst_feat_dict,
                              n_feats=n_feats_used)
    train_cen_X = build_data(data_matrix=train_cen_encoded_matrix,
                             inst_list=train_cen_inst_list,
                             inst_feat_dict=inst_feat_dict,
                             n_feats=n_feats_used)
    test_cen_X = build_data(data_matrix=test_cen_encoded_matrix,
                            inst_list=test_cen_inst_list,
                            inst_feat_dict=inst_feat_dict,
                            n_feats=n_feats_used)

    return train_uncen_X, train_uncen_y, test_uncen_X, test_uncen_y, train_cen_X, test_cen_X, train_cen_y, test_cen_y

def train_model(train_X, train_y, test_X, model, weights=None):
    if weights is None:
        weights = []
    start = time.time()
    m = model
    if weights==[]:
        m.fit(train_X, train_y)
    else:
        m.fit(train_X, train_y, sample_weight=weights)
    dur = time.time() - start
    logging.info("Training model took %s" % dur)

    pred = m.predict(test_X)
    return dur, pred

def get_scenario(scenario, src_dir):
    """
    This makes it easier to get the data for the experiments
    """
    tuning_scenario = ''
    performance_file = ''
    config_file = ''
    pcs_file = ''
    feature_file = ''
    if scenario == 0:
        tuning_scenario = 'SPEAR-SWV' #'SPEAR-swv-al'b
        performance_file = os.path.join(src_dir,'SAT','1000samples-SPEAR-SWV-all604inst-results.txt')
        config_file = os.path.join(src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'SAT','spear-params.txt')
        feature_file = os.path.join(src_dir,'SAT','SWV-feat.csv')
    elif scenario == 1:
        tuning_scenario = 'SPEAR-IBM' #'SPEAR-ibm-al'
        performance_file = os.path.join(src_dir,'SAT','1000samples-SPEAR-IBM-all765inst-results.txt')
        config_file = os.path.join(src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'SAT','spear-params.txt')
        feature_file = os.path.join(src_dir,'SAT','IBM-ALL-feat.csv')
    elif scenario == 2:
        tuning_scenario = 'SPEAR-SWV-IBM' #'SPEAR-ibm-swv-al'
        performance_file = os.path.join(src_dir,'SAT','1000samples-SPEAR-IBM-SWV-results.txt')
        config_file = os.path.join(src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'SAT','spear-params.txt')
        feature_file = os.path.join(src_dir,'SAT','IBM-SWV-feat.csv')
    elif scenario == 3:
        tuning_scenario = 'CPLEX-CRR' #'CPLEX12-cat-CORLAT-REG-RCW' # memory error
        performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-CORLAT-REG-RCW-results.txt')
        config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(src_dir,'MIP','CORLAT-REG-RCW-features.csv')
    elif scenario == 4:
        tuning_scenario = 'CPLEX-CR' #'CPLEX12-cat-CORLAT-REG' # memory error
        performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-CORLAT-REG-results.txt')
        config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(src_dir,'MIP','CORLAT-REG-features.csv')
    elif scenario == 5:
        tuning_scenario = 'CPLEX-RCW' # memory error
        performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-RCW-990train-990test-results.txt')
        config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(src_dir,'MIP','RCW-train_test-features-withfilename.csv')
    elif scenario == 6:
        tuning_scenario = 'CPLEX-REG' # memory error
        performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-CATS_REG-1000train-1000test-results.txt')
        config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(src_dir,'MIP','REG-train_test-features-withfilename.csv')
    elif scenario == 7:
        tuning_scenario = 'CPLEX-CORLAT'
        performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-CORLAT-train_test_inst-results.txt')
        config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(src_dir,'MIP','CORLAT-train_test-features-withfilename.csv')
    elif scenario == 8:
        tuning_scenario = 'CPLEX-BIGMIX'
        performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-BIGMIX-all1510inst-results.txt')
        config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
        pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
        feature_file = os.path.join(src_dir,'MIP','BIGMIX-train_test-features-withfilename.csv')

    return performance_file, config_file, pcs_file, feature_file
    
def eval_model(name, train_X, log10_train_y, test_X, log10_test_y, model,
               weights=None):
    if weights is None:
        weights = []
    logging.info("###############################################")
    logging.info("## Evaluating %s" % name)
    logging.info("### Uncensored")
    logging.info("Start training (%s) and predicting with non censored data" % str(train_X.shape))
    
    dur, log10_pred = train_model(train_X=train_X, train_y=log10_train_y, test_X=test_X, model=model,weights=weights)
    rmse = numpy.sqrt(mean_squared_error(y_true=log10_test_y, y_pred=log10_pred))
    
    logging.info("RMSE: %s" % rmse)

    return dur, log10_pred, rmse
    
def main():
    tuningS={'SPEAR-SWV': 0,
             'SPEAR-IBM': 1,
             'SPEAR-SWV-IBM': 2,
             'CPLEX-CRR': 3,
             'CPLEX-CR': 4,
             'CPLEX-RCW': 5,
             'CPLEX-REG': 6,
             'CPLEX-CORLAT': 7,
             'CPLEX-BIGMIX': 8}

    parser = argparse.ArgumentParser(description='Evaluate a model on data')
    parser.add_argument('-s', '--scenario', dest='scenario', default=1, required=True, choices=sorted(tuningS.keys()),
                        type=str, help='which scenario')
    parser.add_argument('-n', '--normalize', dest='normalize', action='store_true',
                        help='Normalize features')
    parser.add_argument('-c', '--cutoff', dest='cutoff', default=300, type=float)
    parser.add_argument('-d', '--dir', dest='src_dir', default = '../../data/matlab_data')
    parser.add_argument('--save', dest='save', required=True,
                        help="Where to save predictions")
    parser.add_argument('-a', '--all', action='store_true', dest='all', help='Use all datapoints')
    parser.add_argument('-p', '--par', type=int, default=1, dest='multiplyer', help='PARX multiplyer')
    parser.add_argument('-w', '--weighted', action='store_true', dest='weighted', help='Weigh samples')
    args, unknown = parser.parse_known_args()
    
    csv_file, config, pcs_file, feature_file = get_scenario(tuningS[args.scenario], args.src_dir)
    
    logging.basicConfig(level=logging.INFO)

#   Tests whether file exists
    for f in (pcs_file, feature_file):
        assert os.path.isfile(f), "%s is not a directory" % f

    if not os.path.isdir(os.path.dirname(args.save)):
        os.makedirs(args.save)
        
    assert os.path.isdir(os.path.dirname(args.save)), "%s is not a directory" % args.save
    
    assert os.path.isfile(config), "%s is not a file" % config

    pred_list = list()
    test_list = list()
    
    models={'rrf': ('ML4AAD', None, None),
            'rf' : model_dict['rf']
           }
   
    hotOrNot=[True,False]
    
    indices = None
    used_indices=[]
    
    for oneHot in hotOrNot: # first run everything for sklearn, then for stephans
        logging.info('Loading data')
        logging.info('One-hot-encode: %s'%str(oneHot))
        encoded_matrix, inst_feat_dict, inst_list, n_feats_used, perf_list, suc_list, cen_list, best_perf = \
            loadMatlabData(pcs_file=pcs_file, feature_file=feature_file,
                           performance_file=csv_file, config_file=config,
                           normalize_features=args.normalize, max_y=args.cutoff, oneHot=oneHot)
        del suc_list, best_perf
                           
#       Split data in four parts
        if indices is None:
            indices = partition_data(data_matrix=encoded_matrix,
                                     inst_list=inst_list, rs=RNG)
        
        if oneHot:
            name = models['rf'][0]
            model = models['rf'][1]()
        else:
            model = ExternalRFR(cs=ConfigSpace(pcs_file), n_feats=n_feats_used)
            name = models['rrf'][0]
                
        logging.info("Use %s as EPM model" % name)
        
#       Now use diagonal pairs of datasets    
        for c_idx in (False, True):
            for i_idx in (False, True):
                logging.info("### configs %d, instances %d" % (int(c_idx), int(i_idx)))
                if len(used_indices) < 4:
                    train_idx = indices[int(c_idx)][int(i_idx)]
                    test_idx = indices[int(not c_idx)][int(not i_idx)]
                    if not args.all:
                        train_idx = numpy.random.permutation(train_idx[:10000])
                        test_idx = numpy.random.permutation(test_idx[:10000])
                    used_indices.append((train_idx,test_idx))
                else:
                    at = int(str(int(c_idx))+str(int(i_idx)),base=2)
                    train_idx=used_indices[at][0]
                    test_idx=used_indices[at][1]
                    used_indices.append((train_idx,test_idx))
                
                # See evaluate_regression_new_instances
                train_X, train_y, test_X, test_y, train_cen_X, test_cen_X, train_cen_y, test_cen_y = \
                        preprocess_data(train_idx, test_idx, encoded_matrix, inst_list, perf_list, cen_list, inst_feat_dict, n_feats_used)
                        
                assert len(test_cen_X) == 0
                assert len(train_cen_X) == 0
                assert len(test_cen_y) == 0
                assert len(train_cen_y) == 0
                
                logging.info("Using logscale for train")
                tmp = numpy.array(train_y)
                tmp[tmp>=args.cutoff] = args.cutoff*args.multiplyer
                log10_train_y = numpy.log10(tmp).astype(numpy.float32)
                log10_test_y = numpy.log10(numpy.array(test_y)).astype(numpy.float32)
                test_list.append((test_X, test_y))
                
                weights=[]
                if args.weighted and not name == 'ML4AAD':
                    weights=numpy.ones(tmp.shape)
                    weights[tmp==args.cutoff*args.multiplyer]=10
                
                dur, log10_pred, rmse=eval_model(name, train_X, log10_train_y, test_X, log10_test_y, model,weights=weights)
                
                pred_list.append(numpy.power(10, log10_pred))

    if args.save is not None:
        fh = open(args.save + "_comparison.pkl", 'w')
        pickle.dump((pred_list, test_list
                     ), fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.close()

if __name__ == "__main__":
    main()
