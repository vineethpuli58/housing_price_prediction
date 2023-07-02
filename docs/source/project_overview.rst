================
Project Overview
================



The library contains code for the housing price prediction problem. These templates can be readily set up
in your local environment and used for modelling the median house value on given housing data. 

Approach
========

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

Steps performed
---------------

 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.


Evaluation
==========

The project is created based on the regression code templates archive. The housing price prediction problem is modelled using linear regression,
decision tree and random forest. The models were compared based on their performance and the best model with appropriate features is selcted and the reports on the same were generated.


Architecture
============

The major portion of the project is structured in three notebooks under the `notebooks/reference` folder in the main project folder. 

- Conda environment is created by changing the envoroniment name in the tasks.py
- Customized function `download_data` is defined in the `src/ta_lib/housing/fetch_data` for downloading the data in the `data/raw` folder.
- Parameters for the random forest and the grid search cv are provided throuh the `local.yaml` file in `notebooks/reference/conf/model_catalog`.
- A custom transformer is added in `src/ta_lib/housing/custom_transform` to extract new features for training. The custom transformer supports both forward and backward transform. The documentaion of the new functions is rendered in the api-guide.
- An automated `complexity-score` calculation task is added to `task.py`. The complexity score is evaluated on jupyter-notebooks and the hpp scripts. 
    
