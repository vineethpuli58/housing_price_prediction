===============
Data Processing
===============


Data Cleaning 
-------------

Document key data cleaning steps such as column name & datatype standardizations, discarded columns or filtered out data etc..

Data Consolidation
------------------

Since merging is a critical step and often leads to erroneous models/reruns, document the decisions taken while merging such as type of join, actions in cases of cardinality not as expected etc..


Imputation
----------

Describe the imputation rules and rational behind such rules. For example, filling missing gender with "unknown".

Outlier treatment
-----------------
- The outliers have been treated and replaced with ``mean`` value of column.
- A Custom Transformer is used to handle outliers. It is not included as part of the pipeline as outliers handling are optional for test data
- An option to either drop or cap the outliers can be passed during the transform call
- If we want to treat outliers for some columns them we can pass cols argument to the Transformer
- This will go into production code


Feature transformations
-----------------------
- Commonly target encoding is done for categorical variables with too many levels.
- We also group sparse levels. For fewer levels one hot encoding/label encoding is preferred.
- If there is one dominant level, we can use binary encoding.
- This will go into production code

