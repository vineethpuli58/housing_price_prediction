import optuna
import os.path as op
import sklearn # noqa
from ta_lib.housing import data_prep as dp
from optuna.integration.mlflow import MLflowCallback
from ta_lib.core.api import (
    create_context, load_dataset
)
DEFAULT_CONFIG_PATH = op.abspath(__file__ + '/../../../../production/conf/config.yml')
context = create_context(DEFAULT_CONFIG_PATH)
inp_args = context.config["hyperparam_catalog"]["hyperparam"]


def objective(trial, algorithm):
    """Build the model and evaluate the test process.

    function loads,transforms and trains data to create required model.

    Parameters
    ----------
    trial : an optuna instance
    algorithm : ML regression and classiifcation model
    Returns
    ----------
    metrics : accuracy and best hyperparameters
    """
    train_X = load_dataset(context, 'processed/housing/features')
    train_y = load_dataset(context, 'processed/housing/target')
    housing_prepared = dp.data_prep(train_X)
    params = inp_args[algorithm]["params"]
    param_inp = {}
    for param, value in params.items():
        if type(value[0]) == int:
            param_inp[param] = trial.suggest_int(param, value[0], value[1])
        else:
            param_inp[param] = trial.suggest_float(param, value[0], value[1])
    clf = eval(inp_args[algorithm]["model_import"])(**param_inp)
    clf.fit(housing_prepared, train_y)
    y_pred = clf.predict(housing_prepared)
    metric = eval("sklearn." + inp_args[algorithm]["metric"])(train_y, y_pred)
    return metric


# def func(trial,algorithm):
#     return objective(trial, algorithm)


def tune_param(algorithm):
    """Tuning Parameters for Maximum accuracy.

    function returns best accuracy with optimized hyperparameters.

    Parameters
    ----------
    algorithm : ML regression and classiifcation model
    Returns
    ----------
    metrics : accuracy and best hyperparameters
    """
    HERE = op.dirname(op.abspath(__file__))
    mlrun_path = op.join(HERE, "..", "..", "..", "mlruns")
    mlflc = MLflowCallback(
        tracking_uri=mlrun_path,
        metric_name=inp_args[algorithm]["metric"][8:],
        mlflow_kwargs={"nested": True})
    study = optuna.create_study(
        study_name=inp_args[algorithm]["study_name"],
        direction=inp_args[algorithm]["metric_direction"],
    )
    func = lambda trial : objective(trial, algorithm) # noqa
    study.optimize(
        func,
        n_trials=inp_args[algorithm]["n_trails"],
        callbacks=[mlflc],
    )
    trial = study.best_trial
    return {
        inp_args[algorithm]["metric"][8:]: trial.value,
        "Best hyperparameters": trial.params,
    }
