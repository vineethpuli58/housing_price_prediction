=======================
CombinedAttributesAdder 
=======================

A housing package has been created to download and store data . Apply custom_transform  to create more columns as required by problem statement

Description
------------------------------------------------
The class ``combinedattributesadder`` adds three columns as ``rooms_per_household``, ``bedrooms_per_room``, ``population_per_household`` which has been calculated using other columns from dataframe.
Both forward and backward transform has been applied to create the required columns and remove them if required.

The class has been called and used in ``data_prep.py`` to create extra attrributes.The ``data_prep.py`` has been called in model_experimenting.py to create the required transformed dataframe.

=====================
Hyperparameter tuning
=====================

The hyperparameter tuning is done using optuna. Optuna is an open-source Python library for hyperparameter optimization, which is the process of finding the optimal values for the parameters of a machine learning model. Optuna provides a simple and intuitive interface for defining a search space of hyperparameters, specifying an objective function to be optimized, and running a search algorithm to find the best set of hyperparameters. It supports a wide range of search algorithms, including grid search, random search, and Bayesian optimization, and it also includes advanced features such as pruning and parallel execution to improve the efficiency of the search process. Optuna can be used with a variety of machine learning frameworks, including TensorFlow, PyTorch, Scikit-learn, and XGBoost, making it a versatile tool for optimizing machine learning models.

This section describes how to input parameters through config file. The config file is located in ``production/conf/hyperparam_catalog/hyperparam.yml``.
The module can be imported using ``from src.ta_lib.hyperparam_tuning import Hyperparam``. Below is an example of ``hyperparam.yml``

.. literalinclude:: ../../../production/conf/hyperparam_catalog/hyperparam.yml
    :language: yaml
    :caption: hyperparam_catalog configuration

Points to be noted for updating the config file.
------------------------------------------------

1. The name of the model should be the same name as parameter to the ``Hyperparam(algorithm="<model_name>")`` class
2. ``study_name`` assigns the name for the ``mlflow`` experiment.
3. ``n_trails`` should be an integer for the number of trails for optimizing.
4. ``model_import`` is the entire import statement for the model. For example to use ``XGBRegressor()``, the parameter should be ``xgboost.XGBRegressor``
5. ``metric_direction`` is the direction on which we want to optimize Hyperparameter to optimize on **R2-value** the parameter should be ``metric.r2_score``
6. The hyperparameters for a specific algorithm has to be updated in the ``params`` key. It's important to note that, the parameter key should be same as the hyperparameter actual name. The hyperparameter ranges should be provided in a list. 
   
Examples
--------   
.. code-block:: python

    from ta_lib.hyperparam_tuning.api import  tune_param

    tuner=tune_param(algorithm)