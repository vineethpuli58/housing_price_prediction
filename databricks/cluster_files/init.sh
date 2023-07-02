#!/bin/bash
# Go to DBFS path where CT is uploaded
cd /dbfs/code_templates/regression-py/
# Install additional dependencies for Databricks
/databricks/python/bin/pip install -r databricks/cluster_files/requirements_runtime10.5ML.txt
# Installing Code-Templates
/databricks/python/bin/pip install -e .