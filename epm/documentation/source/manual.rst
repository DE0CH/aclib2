Manual
======

.. role:: bash(code)
    :language: bash
    
In the following we will show how to use an **Empirical Performance Model** (*EPM*).

.. _quick:

Quick Start
-----------

The use of EPM is demonstrated using a minimal example to get you up and running.

The download containing the EPM also contains the example. It can be found in EPM/documentation/example.

The example uses 8 unique configurations, 2 unique instances with 18 runs of which 3 are censored.
In this example only two variables are used for the configurations. One of the variables is categorical whilst the other variable is continuous.
They are given in the performance_example.csv file. To make it easier to see the unique configurations we provided the formatted_performance_example.csv file.
In this file the configurations are ordered in a way that makes it easier to see the parameter settings. 

In order to evaluate the given example, download the example data, navigate to the EPM/scripts/evaluate directory and use the following command:
    >>> python evaluate_regression_new_instances.py --pcs example_location/parameter_configuration_space_example.pcs --features example_location/example_features.csv
        --csv example_location/performance_example.csv --model rf --log --cutoff 300 --save example_save_location/result

This will impute the censored data and fit the Random Forest to make predictions about the
runtime.

The data are split into four disjoint sets containing configurations and the corresponding instances. These sets are used
for training and testing respectively. We enumerate the four sets as follows: 00, 01, 10, 11.
Training and Testing pairs are (00, 11) and (01, 10). This means, when we train on set 00, we test on set 11 and vice versa. Sets 00 and 11 contain different configurations and instances. (Likewise for sets 10 and 01)

For plotting the result in a scatter plot, first navigate to the EPM/scripts/plotting folder, then use the following command:
    >>> python scatter_regression_results.py --pkl example_save_location/result_predicitons.pkl --save example_save_location/result

where example_save_location is the path where you saved the .pkl file (the same location as --save example_save_location/result in the evaluation command).

The resulting scatter plot will look like this:

.. image:: ../../example/result_scatter.png
    :width: 800px
    :align: center
    :height: 600px

The first row shows the results without imputing the censored data and the second row shows the results with imputing the censored data.

On the left we trained on set 00 and tested on set 11, in the next image set 01 was used for training and set 10 for testing and so on.

In the given example only three values had to be imputed. In the image above the third set contained no data to impute.

Evaluate a model on data
------------------------

To evaluate a model on data use the following command:
    >>> python evaluate_regression_new_instances.py --pcs pcs_file --csv csv_file --features feature_file --model rf --cutoff [real>0] --save save_path

The shown arguments are necessary for the EPM in order to evaluate the data.

The specification of an algorithms parameters and their possible ranges is done in the **Parameter Configuration Space (PCS)** file. We use SMAC's PCS file format. This format allows one to declare parameters as real, integer, ordinal and categorical, as well as conditional parameters that are only active dependent on other parameters. For an in-depth manual on defining the parameter configuration space please refer to the SMAC `manual <http://www.cs.ubc.ca/labs/beta/Projects/SMAC/#software>`_.

==== ========= ======= =============================
Name   Range   Default
==== ========= ======= =============================
 x1   {0,1,2}    [1]   # Categorical
 x2  [0.0,1.0]  [0.5]  # Real
 x3    [0,10]    [6]   # Integer
==== ========= ======= =============================

The **csv_file** contains information about the performance of a specific configuration on a specific instance. As the name indicates, the values are separated by commas. The following table gives an example using the PCS shown above.

======== ==== ======= =========== ========= ============== =====
Instance seed status  performance config                    
======== ==== ======= =========== ========= ============== =====
 inst_1   0     SAT    0.9102993   -x1=2    -x2=0.03929912 -x3=3
 inst_2   1    UNSAT   9.9922919   -x2=0.95                     
 inst_1   1   TIMEOUT  2.9923882   -x1=0    -x3=9               
 inst_1   2     ...       ...       ...         ...        ...
  ...    ...    ...       ...       ...         ...        ...
======== ==== ======= =========== ========= ============== =====

The first column gives the name of the instance. This is used later together with the feature file to determine all unique instances. In the example above we only show two instances.

The seed currently serves no purpose and can be set to any arbitrary value.

The third column indicates the status of an algorithm run. A run is counted as *successful* if the status is either SAT, UNSAT or SUCCESSFUL. If the status is CRASHED or TIMEOUT a run is counted as *unsuccessful*. Lastly a run is counted as *censored* if the status is TIMEOUT and the performance is smaller than the value specified using *cutoff*, otherwise it is not counted as censored and only counted as unsuccessful.

An algorithms performance (e.g. runtime) is given in the fourth column.

The fifth column indicates the start of the configuration of the algorithm. The configuration goes on in the following columns, depending on how many parameters have to be set per configuration. **The - in front of the parameters are necessary for the parser to be able to parse the data**

In the first row you can see that *x1* is set to 2, *x2* to 0.03929912 and *x3* is set to 3.
In the second row you can see that only *x2* is set to a value. This means that for that specific configuration *x1* takes its default value and *x3* also takes its default value.
Similarly in row three *x2* is not set and its default value will be used.

The **feature file** (also a comma-separated-values file) contains information about all instances and takes the following form:

======== ========== === =========
Instance  Feature 1 ... Feature n
======== ========== === =========
 inst_1     123     ...    0.5
 inst_2    19934    ...    0.1
   ...      ...     ...    ...
 inst_m     234     ...    0.9
======== ========== === =========

where the features of an instance could look something like this:

============= ========= ============ ===== ======== =========== ============== =============== ==================
INSTANCE_NAME nvarsOrig nclausesOrig nvars nclauses reducedVars reducedClauses Pre-featuretime vars-clauses-ratio
============= ========= ============ ===== ======== =========== ============== =============== ==================

In order to determine which type of model has to be fit the **model** parameter has to be set. **rf** corresponds to Random Forest and is currently the only viable option.

**cutoff** awaits a value greater 0. It is the cutoff time that is used to determine if a run is censored or not.

Lastly, the **save** parameter specifies where the resulting .pkl file will be saved. If you specify it like */home/user/doucuments/xyz* the .pkl file will be saved in the users documents folder with the name *xyz* and _predictions.pkl will be appended to the given name. I.e. specifying the save_path as */home/user/documents/algo1* will save *algo1_predictions.pkl* in the users documents folder.

This .pkl file can be used to plot the results. (See :ref:`visu`)

Additionally to the required arguments, the EPM takes optional arguments:

* -h, --help
* --instances
* --log

-h/--help prints the usage.

--instances takes an additional file as input. This file specifies which instances to use with the EPM. If this argument is not used, all instances in the feature file/csv_file will be used.

The instance file only contains the names of the instances to used with one name per row.

--log puts the performance data on the log scale. We suggest using this flag when the performance to predict is an algorithms runtime. Previous work dealing with runtime prediction
found that logarithmic transformations improve a models quality.

.. _visu:

Visualization
-------------

To plot the evaluated data first navigate to the EPM/scripts/plotting folder and then use the following command:
    >>> python scatter_regression_results.py --pkl pkl_file --save save_path

For this command both arguments are required.

To plot the results obtained with the EPM in the previous section, **pkl_file** should be the path you specified using --save during the evaluation step.
Previously we used */home/user/documents/algo1* as example. Since the evaluation step will append _predictions.pkl to the specified name pkl_file should be */home/user/documents/algo1_predictions.pkl*

The **--save** argument again takes a path that specifies where to save the resulting scatter plot. Again we use */home/user/documetns/algo1* as example. The resulting plot will be saved
as png in the users documents folder with the name *algo1_scatter.png*

The script has one optional argument **--max**. Values that are greater than the value specified by --max will be replaced by that value for plotting.

The resulting scatter plot will look something like

.. image:: ../../example/_scatter.png
    :width: 800px
    :align: center
    :height: 600px

As previously described in :ref:`quick` the first row contains the results without the imputed data, the second row contains the results for the imputed data.
In the bottom left corner the error metrics (Root Mean Square Error and the Spearman Correlation Coefficient) are shown for both the imputed and not imputed data.)
