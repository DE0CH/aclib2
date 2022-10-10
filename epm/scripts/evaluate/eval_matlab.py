import os
import argparse
from subprocess import call

tuningS={'SPEAR-SWV': 0,
         'SPEAR-IBM': 1,
         'SPEAR-SWV-IBM': 2,
         'CPLEX-CRR': 3,
         'CPLEX-CR': 4,
         'CPLEX-RCW': 5,
         'CPLEX-REG': 6,
         'CPLEX-CORLAT': 7,
         'CPLEX-BIGMIX': 8,
         'TEST': 9}

parser = argparse.ArgumentParser(description='Evaluate a model on data')
parser.add_argument('-s', dest='s', default=1, required=True, choices=sorted(tuningS.keys()),
                    type=str, help='which scenario')
parser.add_argument('-n', dest='normalize', action='store_true',
                    help='Normalize features')
parser.add_argument('-m', dest='model', default='rf')
parser.add_argument('-c', dest='cutoff', default='1')
parser.add_argument('-k', dest='captype', default='fixed')
parser.add_argument('-a', dest='ac', action='store_true', help='censor data')
parser.add_argument('-d', dest='src_dir', default = '../../data/matlab_data')
parser.add_argument('-t', dest='cap', default = '1')
                    
args, unknown = parser.parse_known_args()

tuning_scenario = ''
performance_file = ''
config_file = ''
pcs_file = ''
feature_file = ''
src_dir = args.src_dir

if tuningS[args.s] == 0:
    tuning_scenario = 'SPEAR-SWV' #'SPEAR-swv-al'b
    performance_file = os.path.join(src_dir,'SAT','1000samples-SPEAR-SWV-all604inst-results.txt')
    config_file = os.path.join(src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'SAT','spear-params.txt')
    feature_file = os.path.join(src_dir,'SAT','SWV-feat.csv')
elif tuningS[args.s] == 1:
    tuning_scenario = 'SPEAR-IBM' #'SPEAR-ibm-al'
    performance_file = os.path.join(src_dir,'SAT','1000samples-SPEAR-IBM-all765inst-results.txt')
    config_file = os.path.join(src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'SAT','spear-params.txt')
    feature_file = os.path.join(src_dir,'SAT','IBM-ALL-feat.csv')
elif tuningS[args.s] == 2:
    tuning_scenario = 'SPEAR-SWV-IBM' #'SPEAR-ibm-swv-al'
    performance_file = os.path.join(src_dir,'SAT','1000samples-SPEAR-IBM-SWV-results.txt')
    config_file = os.path.join(src_dir,'SAT','1000samples-algospear1.2.1.1-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'SAT','spear-params.txt')
    feature_file = os.path.join(src_dir,'SAT','IBM-SWV-feat.csv')
elif tuningS[args.s] == 3:
    tuning_scenario = 'CPLEX-CRR' #'CPLEX12-cat-CORLAT-REG-RCW' # memory error
    performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-CORLAT-REG-RCW-results.txt')
    config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
    feature_file = os.path.join(src_dir,'MIP','CORLAT-REG-RCW-features.csv')
elif tuningS[args.s] == 4:
    tuning_scenario = 'CPLEX-CR' #'CPLEX12-cat-CORLAT-REG' # memory error
    performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-CORLAT-REG-results.txt')
    config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
    feature_file = os.path.join(src_dir,'MIP','CORLAT-REG-features.csv')
elif tuningS[args.s] == 5:
    tuning_scenario = 'CPLEX-RCW' # memory error
    performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-RCW-990train-990test-results.txt')
    config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
    feature_file = os.path.join(src_dir,'MIP','RCW-train_test-features-withfilename.csv')
elif tuningS[args.s] == 6:
    tuning_scenario = 'CPLEX-REG' # memory error
    performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-CATS_REG-1000train-1000test-results.txt')
    config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
    feature_file = os.path.join(src_dir,'MIP','REG-train_test-features-withfilename.csv')
elif tuningS[args.s] == 7:
    tuning_scenario = 'CPLEX-CORLAT'
    performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-CORLAT-train_test_inst-results.txt')
    config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
    feature_file = os.path.join(src_dir,'MIP','CORLAT-train_test-features-withfilename.csv')
elif tuningS[args.s] == 8:
    tuning_scenario = 'CPLEX-BIGMIX'
    performance_file = os.path.join(src_dir,'MIP','1000samples-CPLEX-BIGMIX-all1510inst-results.txt')
    config_file = os.path.join(src_dir,'MIP','1000samples-algocplex12-milp-runobjruntime-overallobjmean10-runs1000-time300.0-length2147483647_0.txt')
    pcs_file = os.path.join(src_dir,'MIP','cplex12-params-CPAIOR-space.txt')
    feature_file = os.path.join(src_dir,'MIP','BIGMIX-train_test-features-withfilename.csv')
elif tuningS[args.s] == 9:
    tuning_scenario = 'TEST'
    src_dir='/home/biedenka/git/epm/documentation/example'
    performance_file = os.path.join(src_dir, 'performance_example.csv')
    config_file = None
    pcs_file = os.path.join(src_dir,'parameter_configuration_space_example.pcs')
    feature_file = os.path.join(src_dir,'example_features.csv')

to_join = ["python",'evaluate_regression_new_instances.py', '--csv', performance_file, '--pcs', pcs_file, '--features', feature_file]

to_save = '%s/%s'%(src_dir, tuning_scenario)

if config_file != None:
    to_join += ['--config', config_file]

to_join+=['--cutoff', args.cutoff]

if args.normalize:
   norm = '--normalize'
   to_join += [norm]

if args.ac:
    ac = '--ac'
    to_join += [ac, '--cap', args.cap, '--captype', args.captype]
    to_save += '_captime_%ssec_captype_%s'%(args.cap,args.captype)

to_join += ['--model', args.model, '--save', to_save]

print(' '.join(to_join))
