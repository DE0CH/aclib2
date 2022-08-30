#Algorithm Configuration Library 2.0 - Surrogates

**This branch offers benchmarks based on empirical performance models (EPMs)**

## What are EPMs?
Empirical performance models are regression models that characterize a given
algorithm's performance across problem instances and/or parameter settings.
These models can predict the performance of algorithms on previously unseen
input, including previously unseen problem instances, previously untested
parameter settings, or both and are useful for analyzing of how an algorithm
performs under different conditions, select promising configurations for a
new problem instance, or  surrogate benchmarks.

## What are Surrogate Benchmarks?

To enable an efficient comparison of different configuration algorithms
(including algorithm configuration and hyperparameter optimization)
surrogate benchmarks can be used. We replace the costly evaluation of the
real target algorithm with a prediction of an EPM. With this method we can
reduce the time required to evaluate one configuration to less than one second
and allow extensive, but computationally feasible, empirical comparisons.

## What is in this repository?

This repository contains code to use surrogate benchmarks within AClib2 accompanying the paper:

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

**The code in this repository is updated to work with Python 3, the latest version of [ACLib](https://bitbucket.org/mlindauer/aclib) and [PyRFR](https://github.com/automl/random_forest_run). For the original implementation in Python 2.7 used in the paper, we refer to [here](https://bitbucket.org/mlindauer/epm/get/SIResubmission.zip)**

## How do I use them?

 1. Follow the instructions to use AClib (marked with [AClib]). You don't need to download instances.
 1. Follow further instruction on how to use these benchmarks (marked with [Surrogates])

## [AClib] Install

AClib requires Python >3.5 (implemented under Anaconda 3.5) and Java >8

`pip install -r requirements.txt`

Some target algorithms may have further dependencies.

## [AClib] Requirements for AC procedures

### SMAC

* Java 8

### ParamILS

* ruby 1.9.3 (https://www.ruby-lang.org/de/news/2014/05/16/ruby-1-9-3-p547-released/)

### GGA

* precompiled on Ubuntu 14.04 LTS with gcc -- probably needs to be recompiled

### IRACE

* R (see irace_2.0 README)
* jsonlite package in R (`install.packages('jsonlite')`)

Call the following command to install irace:

`cd configurators/ && make install` 

## [Surrogates]  Install

You now have to install the epm package. We recommend to work from a virtual 
environment (>Python3.5) and install aclib2 and epm in the same Python environment. 
Install the epm package and write the activation command into a file "startup.sh"
 (which will then be activated to evaluate a target function).
```
git clone git@bitbucket.org:mlindauer/epm.git
cd epm
pip install .
# The following depends on your setup
# echo "source virtualEPM/bin/activate" > startup.sh
```

Then you need to download the trained models and unzip them to your aclib2 folder.
For example, to get the epm for the cplex_regions_200 data set:

```
wget https://www.automl.org/wp-content/uploads/2019/10/cplex_regions200.zip
unzip cplex_regions200.zip  -d ./aclib2
```

All available epms are listed at the end of this document.

## Example

### Configuration

To run a scenario call:

`python aclib2/aclib/run.py -s cplex_regions200_surrogate -c SMAC2 -n 2 --env local --startup startup.sh`

It runs the scenario __cplex_regions200__ with 2 independent SMAC (v2) runs using the surrogate benchmark.

### Validation without Workers

To validate the runs (here training and test performance of final incumbent):

`python aclib2/aclib/validate.py -s cplex_regions200_surrogate -c SMAC2 --env local --startup ./startup.sh --set TRAIN+TEST --mode INC --num_func_evals`

### Statistics and Plots

If you have validated your runs, you can run the following command to get some basic statistics and scatter plots:

`python aclib2/aclib/get_evaluation_stats.py` 

This script will look into "." for runs generated with the previous scripts.

If you have validated your runs over time (`--mode TIME`), you can plot the performance of the configurators over time:

`python aclib2/aclib/plot_perf_over_time.py -s cplex_regions200_surrogate`

## Contact

* Marius Lindauer lindauer@cs.uni-freiburg.de
* Katharina Eggensperger eggenspk@cs.uni-freiburg.de (for surrogate-based benchmarks)

## Issue Tracker

https://bitbucket.org/mlindauer/aclib2/issues

## Scenarios

| Scenario 						    	| Domain	| Configurators 			| #Params 	| #Instances 	| Budget 	| wallclock time(*) | EPM models |
| ------------------------------------- | --------- | ------------------------- |:---------:|:-------------:|:---------:|:-----------------:|:----------:|
| clasp-weighted-sequence_surrogate 	| ASP		| SMAC, ROAR, PILS			| 98		| 240/240	 	| 4d	 	| 100min            | [clasp weighted](https://www.automl.org/wp-content/uploads/2019/10/clasp_weighted.zip)
| cplex_rcw_surrogate        		    | MIP 		| SMAC, ROAR, PILS, GGA		| 74		| 1000/1000		| 2d	 	| <10min            | [cplex rcw](https://www.automl.org/wp-content/uploads/2019/10/cplex_rcw.zip)
| cplex_regions200_surrogate		    | MIP 		| SMAC, ROAR, PILS, GGA		| 74		| 1000/1000		| 2d	 	| <10min            | [cplex regions 200](https://www.automl.org/wp-content/uploads/2019/10/cplex_regions200.zip)
| lpg-zenotravel_surrogate      		| Planning 	| SMAC, ROAR, PILS	 		| 67		| 2000/2000		| 2d	 	| 14h               | [lpg-zenotravel](https://www.automl.org/wp-content/uploads/2019/10/lpg_zenotravel.zip)
| lpg-sattelite_surrogate  		        | Planning 	| SMAC, ROAR, PILS	 		| 67		| 2000/2000		| 2d	 	| 14h               | [lpg satellite](https://www.automl.org/wp-content/uploads/2019/10/lpg_satellite.zip)
| probsat_7sat90_surrogate		    	| SAT		| SMAC, ROAR, PILS, GGA?	| 9			| 250/250		| 3h     	| 3h                | [probsat 7sat90](https://www.automl.org/wp-content/uploads/2019/10/probsat_7sat90.zip)
| minisat_randomk3_surrogate	    	| SAT		| SMAC, ROAR, PILS, GGA?	| 10		| 250/250		| 2d     	| <10min            | [minisat randomk3](https://www.automl.org/wp-content/uploads/2019/10/minisat_randomk3.zip)
| lingeling_circuitfuzz_surrogate	   	| SAT		| SMAC, ROAR, PILS, GGA?	| 322		| 250/250		| 2d     	| <10min            | [lingeling_circuitfuzz](https://www.automl.org/wp-content/uploads/2019/10/lingeling_circuitfuzz.zip)
| clasp_queens_surrogate		    	| SAT		| SMAC, ROAR, PILS, GGA?	| 75		| 250/250		| 3h     	| 3h                | [clasp Queens](https://www.automl.org/wp-content/uploads/2019/10/clasp_queens.zip)
| xgboost_covertype_surrogate	        | ML		| SMAC, ROAR, PILS, IRACE	| 11		| 10/1			| 500 runs  | <10 min           | [XGB](https://www.automl.org/wp-content/uploads/2019/10/xgboost_covertype.zip)
| svm_mnist_surrogate			    	| ML		| SMAC, ROAR, PILS, IRACE	| 7			| 10/1			| 500 runs  | <10 min           | [SVM MNIST](https://www.automl.org/wp-content/uploads/2019/10/svm_mnist.zip)
| [not yet] cryptominisat_circuit_fuzz	| SAT		| SMAC, ROAR, PILS, GGA		| 36		| 299/302		| 2d     	| n.A.              | [not yet]
| [not yet] spear_swgcp				    | SAT		| SMAC, ROAR, PILS			| 26		| 1000/2000		| 5h     	| n.A.              | [not yet]
| [not yet] branin (multi-instance)	    | BBOB		| SMAC, ROAR, PILS			| 1			| 76/75			| 1000 runs | n.A.              | [not yet]


(*) ** NOTE ** estimated wallclock time corresponds to running SMAC on the scenario. Running other optimizers may results in a different time.
 The required wallclock time solely depends on the optimizer's overhead and how many configurations are evaluated. Evaluating one configuration on
 one instance takes less than one second.
