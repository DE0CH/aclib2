# EPM

a python library to build empirical performance models on performance data gathered with ACLib.

Status master branch: [ ![Codeship Status for mlindauer/EPM](https://codeship.com/projects/1ddaf800-1e5c-0133-937e-1293daf310d8/status?branch=master)](https://codeship.com/projects/95253)

# What are Empirical Performance Models for Algorithm Configuration

This repository contains code to build empirical performance models (EPM) from data collected by running configurators on [ACLib](https://bitbucket.org/mlindauer/aclib) benchmark scenarios accompanying the paper:


```
@article{eggensperger-ml18a,
  author    = {K. Eggensperger and M. Lindauer and H. Hoos and F. Hutter and K. Leyton-Brown},
  title     = {Efficient Benchmarking of Algorithm Configurators via Model-Based Surrogates},
  journal   = {Machine Learning},
  volume    = {107},
  year      = {2018},
  pages     = {15-41}
}
```

EPMs are regression models that characterize a given algorithms performance across problem instances and/or parameter settings. 
The application we consider here, it to construct cheap-to-evaluate surrogate benchmarks by replacing expensive target algorithm runs with EPM predictions. These benchmark scenarios share the same configuration space as the real benchmark, but are less expensive to run. 
Besided development and unit-testing they can be used to compare configuration algorithms:

The code in this repository is updated to work with Python 3, the latest version of [ACLib](https://bitbucket.org/mlindauer/aclib) and [PyRFR](https://github.com/automl/random_forest_run). For the original implementation in Python 2.7 used in the paper, we refer to [here](https://bitbucket.org/mlindauer/epm/get/SIResubmission.zip)

## Dependencies
  * numpy >= 1.16.3
  * scipy >= 1.3.0
  * nose >= 1.3.4
  * scikit-learn >= 0.21.0
  * matplotlib >= 1.4.3
  * flask >= 1.0.3
  * gunicorn >= 19.9.0
  * [pyrfr](https://github.com/automl/random_forest_run) >= 0.8.0

Optional:

  * matplotlib >= 1.4.3
  * tabulate >= 0.7.5

## How to install
```
git clone git@bitbucket.org:mlindauer/epm.git
cd epm
pip install .
```
optionally you can run tests with ``python setup.py test``

**Note** Some tests may fail as the tests check for the exact same output which will be different for different numpy/scipy/scikit-learn versions.

## How to build EPMs
We provide an example in ``epm/example`` with subsampled toydata for the *CPLEX RCW* scenario. The following guide will walk you through the steps of training an EPM and using is as an benchmark scenario.

**[1] Dataformat**

Besides ``*.json`` files containing performance data, we also require two more file:

* **param.pcs** defines the configuration space with all parameter ranges and potential conditions
* **instance-features.txt** lists instance features for each instance 

**[2] Train an EPM.**

As a machine learning model, we used quantile regression forests as implemented 
in [random_forest_run](https://github.com/automl/random_forest_run):
The next command trains a QRF on the data provided in 
``./CPLEX-RCW-cont_toydata/SMAC.json`` and saves pyrfr-random forrest in the 
files ``pyrfr_model.out.par10.bin`` and ``pyrfr_wrapper.out.par10.bin``. Also,
the used configuration space and the generated instance-feature-dictioniary are
saved for later usage.

The script will load the data and imputes values of right-censored data points 
(runs where we only observed a lower limit on the running time). You can run 
this command from ``examples/``

```
python ../scripts/surrogate/train_surrogate_PAR10.py --cutoff 10000 \
--pcs CPLEX-RCW-cont_toydata/param.pcs --features CPLEX-RCW-cont_toydata/instance-features.txt \
--model rfrq --save ./CPLEX-RCW --par 10 CPLEX-RCW-cont_toydata/SMAC.json
```
*NOTE:* This might take a while

**[3] Use an EPM as a surrogate.**

Next, you can create a daemon and use the trained model as a surrogate benchmark. 
Therefore, we use a gUnicorn deployed flask-application. The script 
``flask_server.py`` contains the server definition, while ``g_unicorn_app.py``
handles the deployment. Requests to the server are handled by the ``flask_woker.py``

Once started, the process runs as a daemon and uses a TCP/IP connection 
to communicate. If no port or ip is specified, it will search for a free port on 
localhost on its own and stores the server credentials in the experiment 
directory (unique for each daemon).

After creating a server, ``monitor_server.py`` starts always a background task 
to monitor the servers current state. If the server is in idle for too long. 
The server will be shut down. 

However, epm is mainly developed to be used with the 
[aclib](https://bitbucket.org/mlindauer/aclib2/src/master/)

```
python -m epm.webserver.g_unicorn_app --pyrfr_wrapper ./pyrfr_wrapper.random.out.par10.pkl \
 --pyrfr_model ./pyrfr_model.random.out.par10.bin --config_space ./config_space.random.out.par10.pcs \ 
 --inst_feat_dict ./inst_feat_dict.random.out.par10.json \
 --dir . --idle_time 60 --pid 123 start
 
 # Check whether daemon is alive
python -m epm.webserver.flask_server --pyrfr_wrapper ./pyrfr_wrapper.random.out.par10.pkl \
 --pyrfr_model ./pyrfr_model.random.out.par10.bin --config_space ./config_space.random.out.par10.pcs \ 
 --inst_feat_dict ./inst_feat_dict.random.out.par10.json \
 --dir . --idle_time 60 --pid 123 status
```

If that worked, you can query it for the performance of a configuration on an instance:
```
python -m epm.webserver.flask_worker <instance_name> <instance_specific_info> \
 <cutoff> <runlength> <seed> [-<param_name> <value]*
```

*Note:* As a default the daemon stops itself after idling for 100s.

*Note:* If the daemon crashed you should check the output in ``./daemon.log``.
To restart you need potentially to delete the ``lock.file``, and the server credentials
 and start the daemon again. We recommend to set the debug argument to True for 
 more detailed error log.

A query would then be for example:
```
python -m epm.webserver.flask_worker instances/mip/data/RCW2/RCW-INSTANCES-2/map1027-s200538-b04-h40-n5.lp 0 10000.0 2147483647 -1 -barrier_algorithm 0 -barrier_crossover 0 -barrier_limits_corrections -1 -barrier_limits_growth 1.0E12 -barrier_ordering 0 -barrier_startalg 1 -emphasis_memory no -emphasis_mip 0 -emphasis_numerical no -feasopt_mode 0 -lpmethod 0 -mip_cuts_cliques 0 -mip_cuts_covers 0 -mip_cuts_disjunctive 0 -mip_cuts_flowcovers 0 -mip_cuts_gomory 0 -mip_cuts_gubcovers 0 -mip_cuts_implied 0 -mip_cuts_mcfcut 0 -mip_cuts_mircut 0 -mip_cuts_pathcut 0 -mip_cuts_zerohalfcut 0 -mip_limits_aggforcut 3 -mip_limits_cutpasses 0 -mip_limits_cutsfactor 4.0 -mip_limits_gomorycand 200 -mip_limits_gomorypass 0 -mip_limits_submipnodelim 500 -mip_ordertype 0 -mip_strategy_backtrack 0.9999 -mip_strategy_bbinterval 7 -mip_strategy_branch 0 -mip_strategy_dive 0 -mip_strategy_file 1 -mip_strategy_fpheur 0 -mip_strategy_heuristicfreq 0 -mip_strategy_lbheur no -mip_strategy_nodeselect 1 -mip_strategy_presolvenode 0 -mip_strategy_probe 0 -mip_strategy_rinsheur 0 -mip_strategy_search 0 -mip_strategy_startalgorithm 0 -mip_strategy_subalgorithm 0 -mip_strategy_variableselect 0 -network_netfind 2 -network_pricing 0 -preprocessing_aggregator -1 -preprocessing_boundstrength -1 -preprocessing_coeffreduce 2 -preprocessing_dependency -1 -preprocessing_dual 0 -preprocessing_fill 10 -preprocessing_linear 1 -preprocessing_numpass -1 -preprocessing_reduce 3 -preprocessing_relax -1 -preprocessing_repeatpresolve -1 -preprocessing_symmetry -1 -read_scale 0 -sifting_algorithm 0 -simplex_crash 1 -simplex_dgradient 0 -simplex_limits_perturbation 0 -simplex_limits_singularity 10 -simplex_perturbation_switch no -simplex_pgradient 0 -simplex_pricing 0 -simplex_refactor 0 -simplex_tolerances_markowitz 0.01 
```
When finished, you can stop the daemon with 

```
python -m epm.webserver.flask_worker \
 --pyrfr_wrapper ./pyrfr_wrapper.random.out.par10.pkl \
 --pyrfr_model ./pyrfr_model.random.out.par10.bin --config_space ./config_space.random.out.par10.pcs \ 
 --inst_feat_dict ./inst_feat_dict.random.out.par10.json \
 --dir . --idle_time 60 --pid 123 stop
```

**[4] Use an EPM as a surrogate benchmark for Algorithm configuration.**

You can now combine the steps above and benchmark algorithm configuration 
methods by 

* (1) Training an EPM 
* (2) run it in the background as a surrogate benchmark and 
* (3) make your configurator to query the EPM instead of running the real 
target algorithm.

**[5] Use our data.**

We provide data on the accompanying website [ml4aad](http://www.ml4aad.org/algorithm-analysis/epms/). 

**[6] Debug Server.**
If no server is started and no log is written, when working on a cluster, 
you could debug the server with the following command:
```
python -m epm.webserver.g_unicorn_app 
```