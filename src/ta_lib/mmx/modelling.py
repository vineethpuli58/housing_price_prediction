# isort: skip_file
# import configparser
import logging
import math
import os
import re
import traceback
from collections import Counter
from datetime import date, datetime

import holoviews as hv
import numpy as np
import pandas as pd
import statsmodels.api as sm
from hvplot import hvPlot
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import BayesFramework.model_config_handler as mch
import BayesFramework.plot_utils as plf
import BayesFramework.Regression as reg
from tigerml.core.reports import create_report
from tigerml.core.scoring import mape, root_mean_squared_error, wmape

hv.extension("bokeh")

# import pdb


# import statsmodels.formula.api as smf


def get_adj_r2(rsq: float, nrows: int, ncols: int):
    """Calculate adjusted r-squared.

    Parameters
    ----------
    rsq: float
        R square value of dataset
    nrows: int
        Number of rows in the dataset
    ncols: int
        Number of columns in the dataset

    Returns
    -------
    float
        Adjusted R square value.
    """
    s = 1 - rsq
    s1 = (nrows - 1) / (nrows - ncols - 1)
    r = 1 - (s * s1)
    return r


def trn_test_split(data: pd.DataFrame, model_config: dict):
    """Test train split.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    model_config: dict
        Model_config from config.yml

    Returns
    -------
    pd.DataFrame
        Train dataset
    pd.DataFrame
        Test dataset
    pd.Series
        Train set’s dependent variable
    pd.Series
        Test set’s dependent variable
    """
    cols = (
        model_config["columns"]["marketing_vars"].copy()
        + model_config["columns"]["base_vars"].copy()
        + model_config["columns"]["categorical_vars"].copy()
        + [model_config["dv"], model_config["time_column"]]
    )
    cols = list(set(cols))
    data = data.loc[:, cols]
    print(cols)
    if model_config["test_train"]["split"] is True:
        if model_config["test_train"]["split_type"] == "random":
            X = data.drop(model_config["dv"], axis=1)
            y = data[model_config["dv"]]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=model_config["test_train"]["test_size"], random_state=42
            )
            return X_train, X_test, y_train, y_test

        elif model_config["test_train"]["split_type"] == "sequential":
            nrows = data.shape[0]
            trows = nrows * model_config["test_train"]["test_size"]
            train = data.iloc[0:trows, :]
            test = data.iloc[trows:, :]
            return (
                train.drop("dv", axis=1),
                test.drop("dv", axis=1),
                train["dv"],
                test["dv"],
            )
    else:
        X_train = data.drop(model_config["dv"], axis=1)
        y_train = data[model_config["dv"]]
        X_test = X_train.copy()
        y_test = y_train.copy()
        return X_train, X_test, y_train, y_test


def scale_variables(
    X_train: pd.DataFrame, X_test: pd.DataFrame, X: pd.DataFrame, model_config: dict
):
    """Min max variable scaling.

    Parameters
    ----------
    X_train: pd.DataFrame
        Train dataset
    X_test: pd.DataFrame
        Test dataset
    X: pd.DataFrame
        Dataset excluding dependent variable
    model_config: dict
        Model_config from config.yml

    Returns
    -------
    X_train: pd.DataFrame
        Scaled train dataset
    X_test: pd.DataFrame
        Scaled Test dataset
    X: pd.DataFrame
        Scaled X dataset
    """
    vars_to_scale = (
        model_config["columns"]["base_vars"] + model_config["columns"]["marketing_vars"]
    )
    vars_to_scale = list(
        set(vars_to_scale) - set(model_config["columns"]["categorical_vars"])
    )
    rng = model_config["scaling"]["col_range"]
    rng = rng.split(",")
    rng = [int(x) for x in rng]
    rng = tuple(rng)
    scaler = MinMaxScaler(feature_range=rng)
    scaler.fit(X_train[vars_to_scale])
    X_train[vars_to_scale] = scaler.transform(X_train[vars_to_scale])
    X_test[vars_to_scale] = scaler.transform(X_test[vars_to_scale])
    X[vars_to_scale] = scaler.transform(X[vars_to_scale])
    return X_train, X_test, X


def log_transformation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y: pd.Series,
    model_config: dict,
):
    """Apply log transformation.

    Parameters
    ----------
    X_train: pd.DataFrame
        Train dataset
    X_test: pd.DataFrame
        Test dataset
    X: pd.DataFrame
        Input Dataset excluding dependent variable
    y_train: pd.Series
        Dependent variable of trainset
    y_test: pd.Series
        Dependent variable of testset
    y: pd.Series
        Dependent variable
    model_config: dict
        Model_config from config.yml

    Returns
    -------
    X_train: pd.DataFrame
        Log transformed train dataset
    X_test: pd.DataFrame
        Log transformed test dataset
    X: pd.DataFrame
        Log transformed input dataset
    y_train: pd.Series
        Log transformed dependent variable of trainset
    y_test: pd.Series
        Log transformed dependent variable of testset
    y: pd.Series
        Log transformed dependent variable
    """
    vars_to_transform = (
        model_config["columns"]["base_vars"].copy()
        + model_config["columns"]["marketing_vars"].copy()
    )
    vars_to_transform = list(
        set(vars_to_transform) - set(model_config["columns"]["categorical_vars"])
    )
    for v in vars_to_transform:
        X_train[v] = np.log(X_train[v] + 1)
        X_test[v] = np.log(X_test[v] + 1)
        X[v] = np.log(X[v] + 1)
    y_train = np.log(y_train + 1)
    y_test = np.log(y_test + 1)
    y = np.log(y + 1)
    return X_train, X_test, X, y_train, y_test, y


def semi_log_transformation(y_train: pd.Series, y_test: pd.Series, y: pd.Series):
    """Apply semi log transformation.

    Parameters
    ----------
    y_train: pd.Series
        Dependent variable of trainset
    y_test: pd.Series
        Dependent variable of testset
    y: pd.Series
        Dependent variable

    Returns
    -------
    y_train: pd.Series
        Log transformed dependent variable of trainset
    y_test: pd.Series
        Log transformed dependent variable of testset
    y: pd.Series
        Log transformed dependent variable
    """
    y_train = np.log(y_train + 1)
    y_test = np.log(y_test + 1)
    y = np.log(y + 1)
    return y_train, y_test, y


def build_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    model_config: dict,
):
    """Build lasso model.

    Parameters
    ----------
    X_train: pd.DataFrame
        Train dataset
    y_train: pd.Series
        Dependent variable of trainset
    X_test: pd.DataFrame
        Test dataset
    y_test: pd.Series
        Dependent variable of testset
    X: pd.DataFrame
        Dataset excluding dependent variable
    y: pd.Series
        Dependent variable
    model_config: dict
        Model_config from config.yml

    Returns
    -------
    coef_df: pd.DataFrame
        Dataframe with Lasso coefficients.
    y_train_pred: np.ndarray
        Predictions for trainset
    y_test_pred: np.ndarray
        Predictions of testset
    y_pred: np.ndarray
        Predictions for X dataset
    """
    # dv = model_config["dv"]
    ts = model_config["time_column"]
    X = X.drop([ts], axis=1)
    model = LassoCV()
    model.fit(X_train, y_train)
    coef_df = pd.DataFrame()
    coef_df["column"] = X_train.columns
    coef_df["beta"] = model.coef_
    i_df = pd.DataFrame()
    i_df["column"] = ["global_intercept"]
    i_df["beta"] = [model.intercept_]
    coef_df = coef_df.append(i_df)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_pred = model.predict(X)
    return coef_df, y_train_pred, y_test_pred, y_pred


def build_bayesian(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    model_config: dict,
    global_config: dict,
    framework_config_df: pd.DataFrame,
    model_config_df: pd.DataFrame,
):
    """Build bayesian model.

    Parameters
    ----------
    X_train: pd.DataFrame
        Train dataset
    y_train: pd.Series
        Dependent variable of trainset
    X_test: pd.DataFrame
        Test dataset
    y_test: pd.Series
        Dependent variable of testset
    X: pd.DataFrame
        Dataset excluding dependent variable
    y: pd.Series
        Dependent variable
    model_config: dict
        Model_config from config.yml
    global_config: dict
        Global_config from config.yml
    framework_config_df : pd.DataFrame
        Dataframe with Bayesian Framework configurations
    model_config_df : pd.DataFrame
        Dataframe with Bayesian model configurations

    Returns
    -------
    beta_df: pd.DataFrame
        Dataframe with beta coefficients
    pred_results_train: np.ndarray
        Predictions for trainset
    pred_results_test: np.ndarray
        Predictions for testset
    pred_results: np.ndarray
        Predictions for X dataset
    """
    train = X_train.copy()
    train[model_config["dv"]] = y
    cols = train.columns.tolist()
    cols_dict = {x.replace(".", "_"): x for x in cols}
    cols_dict["global_intercept"] = "global_intercept"
    cols_dict["sigma_target"] = "sigma_target"
    train.columns = [x.replace(".", "_") for x in cols]
    test = X_test.copy()
    test[model_config["dv"]] = y_test.copy()
    d = X.copy()
    d[model_config["dv"]] = y.copy()
    test.columns = [x.replace(".", "_") for x in test.columns.tolist()]
    d.columns = [x.replace(".", "_") for x in d.columns.tolist()]
    exp_name = "new_run"
    run_name = global_config["run_name"]
    # config_ini_name = ""
    # config_excel = model_config["type_of_model"]["bayes_config_file"]
    # get_config_data = mch.Config(config_ini_name)
    # model_config_df, framework_config_df = get_config_data.get_config(config_excel)
    model_config_df["IDV"] = model_config_df["IDV"].str.replace(".", "_")
    model_config_df["DV"] = model_config_df["DV"].str.replace(".", "_")
    bl = reg.BayesianEstimation(
        train,
        model_config_df,
        framework_config_df,
        experiment_name=exp_name,
        run_name=run_name,
    )
    bl.train()
    bl.summary()
    pl = plf.Plot(bl)
    pl.save_all_plots()
    # Below files are read from Bayesian Framework working directory
    files = os.listdir(model_config["type_of_model"]["bayes_wd"])
    files = [f for f in files if f.split(".")[-1] == "xlsx"]
    paths = [
        os.path.join(model_config["type_of_model"]["bayes_wd"], basename)
        for basename in files
    ]
    req_path = max(paths, key=os.path.getctime)
    print(req_path)
    beta_df = pd.read_excel(req_path, sheet_name="Sheet1")
    pred_results_train, r2score, rms, mapp, ma, wmap = bl.predict(data_pr=train)
    pred_results_test, r2score, rms, mapp, ma, wmap = bl.predict(data_pr=test)
    pred_results, r2score, rms, mapp, ma, wmap = bl.predict(data_pr=d)
    cols = beta_df.columns.tolist()
    cols[0] = "column"
    beta_df.columns = cols
    beta_df["column"] = beta_df["column"].str.replace("fixed_slope_", "")
    beta_df["column"] = beta_df["column"].str.replace("slope_", "")
    beta_df = beta_df.replace({"column": cols_dict})
    # beta_df["column"] = beta_df["column"].map(cols_dict)
    return beta_df, pred_results_train, pred_results_test, pred_results


def get_metrics(
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    y: pd.Series,
    y_pred: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X: pd.DataFrame,
    model_config: dict,
):
    """Generate model metrics.

    Parameters
    ----------
    y_train: pd.Series
        Dependent variable of trainset
    y_train_pred: np.ndarray
        Predictions of trainset
    y_test: pd.Series
        Dependent variable of testset
    y_test_pred: np.ndarray
        Predictions of testset
    y: pd.Series
        Dependent variable
    y_pred: np.ndarray
        Predictions for dependent variable
    X_train: pd.DataFrame
        Train dataset
    X_test: pd.DataFrame
        Test dataset
    X: pd.DataFrame
        Dataset excluding dependent variable
    model_config: dict
        Model_config from config.yml

    Returns
    -------
    pd.DataFrame
        Dataframe with all the model metrics.
    """
    if model_config["type_of_model"]["type"] == "multiplicative":
        y_train = np.exp(y_train)
        y_train_pred = np.exp(y_train_pred)
        y_test = np.exp(y_test)
        y_test_pred = np.exp(y_test_pred)
        y = np.exp(y)
        y_pred = np.exp(y_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rsqu = r2_score(y, y_pred)
    metrics = pd.DataFrame()
    metrics["metric"] = ["rmse", "mape", "wmape", "adjr2", "r2"]
    metrics["train"] = [
        root_mean_squared_error(y_train, y_train_pred),
        (mape(y_train, y_train_pred) * 100),
        (wmape(y_train, y_train_pred) * 100),
        get_adj_r2(train_r2, X_train.shape[0], X_train.shape[1]),
        train_r2,
    ]
    metrics["test"] = [
        root_mean_squared_error(y_test, y_test_pred),
        (mape(y_test, y_test_pred) * 100),
        (wmape(y_test, y_test_pred) * 100),
        "NA",
        test_r2,
    ]
    metrics["full"] = [
        root_mean_squared_error(y, y_pred),
        (mape(y, y_pred) * 100),
        (wmape(y, y_pred) * 100),
        get_adj_r2(rsqu, X.shape[0], X.shape[1]),
        rsqu,
    ]
    return metrics


def get_exp_df(df: pd.DataFrame, model_config: dict):
    """Return exponential values of all variables of a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Input Dataset
    model_config: dict
        Model_config from config.yml

    Returns
    -------
    pd.DataFrame
        Dataframe with the exponential values.
    """
    indicator_cols = model_config["columns"]["categorical_vars"].copy()
    cols = [
        x
        for x in df.columns.tolist()
        if (x not in indicator_cols)
        & (x != model_config["time_column"])
        & (x != model_config["sales_dollars_column"])
    ]
    if model_config["type_of_model"]["transformation"] == "loglog":
        for c in cols:
            df[c] = df[c].apply(lambda x: math.exp(x))
    else:
        dv = model_config["dv"]
        df[dv] = df[dv].apply(lambda x: math.exp(x))
    return df


def subset_columns(data: pd.DataFrame, model_config: dict):
    """Subset columns.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    model_config: dict
        Model_config from config.yml

    Returns
    -------
    pd.DataFrame
        Input Dataset with required columns
    """
    cols = (
        model_config["columns"]["marketing_vars"].copy()
        + model_config["columns"]["base_vars"].copy()
        + model_config["columns"]["categorical_vars"].copy()
        + [
            model_config["dv"],
            model_config["time_column"],
            model_config["sales_dollars_column"],
        ]
    )
    cols = list(set(cols))
    cols = filter(None, cols)
    data = data.loc[:, cols]
    return data


def mixed_lm(df: pd.DataFrame, config: dict):
    """Mixed Linear model.

    Parameters
    ----------
    df: pd.DataFrame
        Input Dataset
    config: dict
        Model_config from config.yml

    Returns
    -------
    fixed_params_df: pd.DataFrame
        Dataframe with fixed variables and their coefficients
    random_params_df: pd.DataFrame
        Dataframe with random variables and their coefficients
    """
    random_effect_cols = config["columns"]["random_effect_cols"]
    fixed_effect_cols = (
        config["columns"]["base_vars"] + config["columns"]["marketing_vars"]
    )
    fixed_effect_cols = list(
        (Counter(fixed_effect_cols) - Counter(random_effect_cols)).elements()
    )
    Y_col = config["dv"]
    group_id_col = config["gv"]
    out_path = config["out_path"]
    date_col = config["time_column"]
    # df['date_sunday'] = pd.to_datetime(df['date_sunday'])
    df.sort_values(by=[config["gv"], config["time_column"]], inplace=True)
    df[Y_col] = df[Y_col].apply(lambda x: np.log(x) if x > 0 else 0)
    # intercept = "global_intercept"
    # date = config["time_column"]
    cols_to_scale = fixed_effect_cols + random_effect_cols
    cols_to_scale = [col for col in cols_to_scale if col is not date_col]
    # create a scaler instance
    scaler = MinMaxScaler()
    # transform the data with scaler
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    endog = df[Y_col]
    groups = df[group_id_col]  # groups
    exog = df[fixed_effect_cols]  # fixed effects
    exog_re = df[random_effect_cols]  # random effects

    # fit Mixed effect model on data
    md = sm.MixedLM(endog=endog, groups=groups, exog=exog, exog_re=exog_re)
    mdf = md.fit()

    # save fixed effect coefficients to disk
    fixed_params_df = pd.DataFrame(mdf.params[fixed_effect_cols])
    fixed_params_df.index.name = "column"
    fixed_params_df.columns = ["beta"]
    fixed_params_df.reset_index(level=0, inplace=True)

    # save random effect coefficients to disk
    random_params_df = pd.DataFrame(mdf.random_effects).T
    random_params_df.index.name = group_id_col
    random_params_df.reset_index(level=0, inplace=True)

    # Saving the coefficients to file:
    wb = wb = f"{out_path}/MixedLM_results.xlsx"
    writer = pd.ExcelWriter(wb)
    fixed_params_df.to_excel(writer, "Fixed_effect_coeff", index=False)
    random_params_df.to_excel(writer, "Random_effect_coeff", index=False)
    final_columns = (
        [group_id_col, config["time_column"]]
        + fixed_effect_cols
        + random_effect_cols
        + [Y_col]
    )
    df[final_columns].to_excel(writer, "IDV_data", index=False)
    writer.save()
    return fixed_params_df, random_params_df


def get_coeff_from_lmer(
    fixed_coeff: pd.DataFrame, random_coeff: pd.DataFrame, config: dict
):
    """Intermediate function to get coefficients of variables.

    Parameters
    ----------
    fixed_coeff: pd.DataFrame
        Coefficients for fixed variables
    random_coeff: pd.DataFrame
        Coefficients for random variables
    config: dict
        Model config from config.yml

    Returns
    -------
    pd.DataFrame
        Coefficients Dataframe
    """
    # gv = config["gv"]
    # random_coeff.drop(gv, axis="columns", inplace=True)
    rand = {}
    for i in config["columns"]["random_effect_cols"]:
        rand[i] = random_coeff[i].mean()
    rand_df = pd.DataFrame()
    rand_df["column"] = rand.keys()
    rand_df["beta"] = rand.values()
    coeff_df = pd.concat([fixed_coeff, rand_df], ignore_index=True)
    return coeff_df


def get_framework_config_data(bayesian_config: dict):
    """Intermediate function to prepare bayesian framework data.

    Parameters
    ----------
    bayesian_config: dict
        Bayesian config from config.yml

    Returns
    -------
    pd.DataFrame
        Dataframe with bayesian framework configuration
    """
    framework_df = pd.DataFrame()
    framework_df["TagName"] = bayesian_config["framework"].keys()
    framework_df["Value"] = bayesian_config["framework"].values()
    return framework_df


def assign_coeff(row: pd.Series, bayesian_config: dict):
    """Intermediate function to assign priors to variables.

    Parameters
    ----------
    row: pd.Series
        Each row of bayesian model config dataframe
    bayesian_config: dict
        Bayesian config from config.yml

    Returns
    -------
    float
        Coefficient of each row variable
    """
    if (
        row["IDV"] in bayesian_config["model"]["expected_coefficients_lt_0"]
        and row["fixed_d_scale_beta"] >= 0
    ):
        return -1
    elif (
        row["IDV"] in bayesian_config["model"]["expected_coefficients_gt_0"]
        or row["fixed_d_scale_beta"] <= 0
    ):
        return 1
    else:
        return row["fixed_d_scale_beta"]


def get_model_config_data(
    model_config: dict, bayesian_config: dict, coef_df: pd.DataFrame
):
    """Intermediate function to prepare bayesian model config.

    Parameters
    ----------
    model_config: dict
        Model config from config.yml
    bayesian_config: dict
        Bayesian config from config.yml
    coef_df: pd.DataFrame
        Coefficients from Lasso

    Returns
    -------
    pd.DataFrame
        Dataframe with bayesian model configurations
    """
    model_df = pd.DataFrame()
    # For Fixed effect
    model_df["IDV"] = coef_df["column"]
    model_df["DV"] = model_config["dv"]
    model_df["Include_IDV"] = 1
    model_df["RandomEffect"] = 0
    fixed_d = bayesian_config["model"]["distribution"]["fixed_cols"][0]
    inter_d = bayesian_config["model"]["distribution"]["intercept_col"][0]
    model_df["fixed_d"] = [
        inter_d if x == "global_intercept" else fixed_d for x in model_df["IDV"]
    ]
    model_df["fixed_d_loc_alpha"] = coef_df["beta"]
    model_df["fixed_d_scale_beta"] = 5
    model_df["fixed_d_scale_beta"] = model_df.apply(
        lambda row: assign_coeff(row, bayesian_config), axis=1
    )
    model_df["fixed_d_scale_beta"] = np.where(
        model_df["fixed_d_scale_beta"] < 0, 1, model_df["fixed_d_scale_beta"]
    )
    model_df["fixed_bijector"] = "Identity"
    if bayesian_config["model"]["type"] == "Fixed_effect":
        cols = [
            "RandomFactor",
            "mu_d",
            "mu_d_loc_alpha",
            "mu_d_scale_beta",
            "sigma_d",
            "sigma_d_loc_alpha",
            "sigma_d_scale_beta",
            "mu_bijector",
            "sigma_bijector",
        ]
        for i in cols:
            model_df[i] = ""
    # For Mixed effect
    else:
        model_df_rnd = pd.DataFrame()
        if bayesian_config["model"]["random_effect_cols"] is not None:
            for i in model_df["IDV"]:
                if i in bayesian_config["model"]["random_effect_cols"]:
                    model_df = model_df.drop(model_df[(model_df.IDV == i)].index)
            model_df_rnd["IDV"] = [
                "intercept_" + str(bayesian_config["model"]["grouping_var"])
            ] + bayesian_config["model"]["random_effect_cols"]
        else:
            model_df_rnd["IDV"] = [
                "intercept_" + str(bayesian_config["model"]["grouping_var"])
            ]
        model_df_rnd["DV"] = model_config["dv"]
        model_df_rnd["Include_IDV"] = 1
        model_df_rnd["RandomEffect"] = 1
        cols_rnd = [
            "fixed_d",
            "fixed_d_loc_alpha",
            "fixed_d_scale_beta",
            "fixed_bijector",
        ]
        for i in cols_rnd:
            model_df_rnd[i] = ""
        model_df_rnd["RandomFactor"] = bayesian_config["model"]["grouping_var"]
        model_df_rnd["mu_d"] = bayesian_config["model"]["distribution"]["rand_cols"][0]
        model_df_rnd["mu_d_loc_alpha"], model_df_rnd["mu_d_scale_beta"] = (
            bayesian_config["model"]["distribution"]["mu_dist_alpha_beta"][0],
            bayesian_config["model"]["distribution"]["mu_dist_alpha_beta"][1],
        )
        model_df_rnd["sigma_d"] = bayesian_config["model"]["distribution"]["rand_cols"][
            1
        ]
        model_df_rnd["sigma_d_loc_alpha"], model_df_rnd["sigma_d_scale_beta"] = (
            bayesian_config["model"]["distribution"]["sigma_dist_alpha_beta"][0],
            bayesian_config["model"]["distribution"]["sigma_dist_alpha_beta"][1],
        )
        model_df_rnd["mu_bijector"], model_df_rnd["sigma_bijector"] = (
            bayesian_config["model"]["distribution"]["bijector"][0],
            bayesian_config["model"]["distribution"]["bijector"][1],
        )

        model_df = model_df.append(model_df_rnd, ignore_index=True)

    # Rearranging config columns sequence
    cols_seq = [
        "DV",
        "IDV",
        "Include_IDV",
        "RandomEffect",
        "RandomFactor",
        "mu_d",
        "mu_d_loc_alpha",
        "mu_d_scale_beta",
        "sigma_d",
        "sigma_d_loc_alpha",
        "sigma_d_scale_beta",
        "mu_bijector",
        "sigma_bijector",
        "fixed_d",
        "fixed_d_loc_alpha",
        "fixed_d_scale_beta",
        "fixed_bijector",
    ]
    model_df = model_df[cols_seq]
    model_df.fillna("", inplace=True)
    return model_df


def spliter(dataframe: pd.DataFrame, grp_var: str, global_config: dict, out_path=""):
    """Intermediate function to split coefficient file for each group.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Coefficient dataframe
    grp_var: str
        Group variable in the data
    global_config: dict
        Global config from config.yml
    out_path: str
        Output path to save expanded file

    Returns
    -------
    str:
        Expanded file name
    """
    try:
        today = date.today()
        d1 = today.strftime("%d-%m-%Y")
        output_file = (
            f'{out_path}/coefficients_df_{global_config["run_name"]}_{d1}.xlsx'
        )
        coef_data = dataframe
        intercept_rows = coef_data["column"].str.contains("intercept_" + grp_var)
        if intercept_rows.any():
            coef_data[["column", "group_var"]] = coef_data["column"].str.split(
                "_" + grp_var, expand=True
            )
            coef_data.loc[intercept_rows, "column"] = "intercept_" + grp_var
            coef_data["group_var"] = coef_data["group_var"].apply(
                lambda x: re.sub(r"\[|]|'", "", x) if x else None
            )
        coef_data.to_excel(output_file, index=False)
        return output_file
    except FileNotFoundError:
        print(traceback.format_exc())
    except Exception:
        print(traceback.format_exc())


def merger(
    coefficient: str,
    model_df: pd.DataFrame,
    framework_df: pd.DataFrame,
    global_config: dict,
    out_path="",
):
    """Intermediate function to merge coefficient and bayesian config.

    Parameters
    ----------
    coefficient: str
        Coefficients file name
    model_df: pd.DataFrame
        Bayesian model config
    framework_df: pd.DataFrame
        Bayesian framework config
    global_config: dict
        Global config from config.yml
    out_path: str
        Output path to save merged file

    Returns
    -------
    str:
        Bayesian config output file name
    """
    try:
        today = date.today()
        d1 = today.strftime("%d-%m-%Y")
        output_file = (
            f'{out_path}/Bayesian_output_{global_config["run_name"]}_{d1}.xlsx'
        )
        conf = model_df
        framework = framework_df
        coef = pd.read_excel(coefficient)
        coef["column"] = coef.column.str.replace(".", "_")
        conf["column"] = conf["IDV"].str.replace(".", "_")
        data = pd.merge(conf, coef, on="column", how="outer")
        random_column = data["RandomFactor"].isnull() | data["RandomFactor"].eq("")
        data.loc[random_column, "fixed_d_loc_alpha"] = data.loc[random_column, "beta"]
        data.loc[~random_column, "mu_d_loc_alpha"] = data.loc[~random_column, "beta"]
        data.drop(["beta", "column"], axis=1, inplace=True)
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            framework.to_excel(writer, sheet_name="Framework", index=False)
            data.to_excel(writer, sheet_name="Model", index=False)
            writer.save()
        return output_file
    except FileNotFoundError:
        print(traceback.format_exc())
    except Exception:
        print(traceback.format_exc())


def get_config_file(bayes_op_framework: pd.DataFrame, bayes_op_model: pd.DataFrame):
    """Intermediate function to get the config params from Bayesian output file.

    Parameters
    ----------
    bayes_op_framework: pd.DataFrame
        Bayesian Framework config from file
    bayes_op_model: pd.DataFrame
        Bayesian model config from file

    Returns
    -------
    model_df: pd.DataFrame
        Bayesian model config
    framework_df: pd.DataFrame
        Bayesian framework config
    """
    framework_df = bayes_op_framework.copy()
    if "group_var" in bayes_op_model.columns:
        bayes_op_model.drop("group_var", axis="columns", inplace=True)
    model_df1 = bayes_op_model.groupby("IDV", as_index=False).agg(
        {"mu_d_loc_alpha": "mean"}
    )
    model_df = pd.merge(
        bayes_op_model, model_df1, left_on=["IDV"], right_on=["IDV"], how="inner"
    )
    model_df.drop(["mu_d_loc_alpha_x"], axis=1, inplace=True)
    model_df.drop_duplicates(inplace=True)
    model_df.rename(columns={"mu_d_loc_alpha_y": "mu_d_loc_alpha"}, inplace=True)
    return model_df, framework_df


def bayesian_model_func(
    data,
    X_train,
    X_test,
    X,
    y_train,
    y_test,
    y,
    model_config,
    bayesian_config,
    global_config,
    out_path="",
):
    gv = bayesian_config["model"]["grouping_var"]
    if gv is not None:
        Xtrain = X_train.drop([gv], axis=1)
        Xtest = X_test.drop([gv], axis=1)
        XX = X.drop([gv], axis=1)
    if len(model_config["columns"]["categorical_vars"]) > 0:
        for i in model_config["columns"]["categorical_vars"]:
            if type(i[0]) != np.number or i == gv:
                continue
            else:
                Xtrain = Xtrain.drop(i, axis="columns")
                Xtest = Xtest.drop(i, axis="columns")
                XX = XX.drop(i, axis="columns")
    else:
        Xtrain = X_train.copy()
        Xtest = X_test.copy()
        XX = X.copy()

    if bayesian_config["lasso_flag"] == 1:
        if bayesian_config["lasso_type"] != "Mixed_effect":
            # LassoCV for priors:
            coef_df2, y_train_pred2, y_test_pred2, y_pred2 = build_lasso(
                Xtrain, y_train, Xtest, y_test, XX, y, model_config
            )
        else:
            # LMER for priors
            fixed_coeff, random_coeff = mixed_lm(data, model_config)
            coef_df2 = get_coeff_from_lmer(fixed_coeff, random_coeff, model_config)

        # Preparing bayesian configurations:
        framework_df = get_framework_config_data(bayesian_config)
        model_df = get_model_config_data(model_config, bayesian_config, coef_df2)
    elif (
        bayesian_config["lasso_flag"] == 0
        and bayesian_config["bayesian_prior_flag"] == 1
    ):
        # Read Bayesian output file and get config params
        bayes_op_model = pd.read_excel(
            bayesian_config["bayesian_prior_file"], sheet_name="Model"
        )
        bayes_op_framework = pd.read_excel(
            bayesian_config["bayesian_prior_file"], sheet_name="Framework"
        )
        model_df, framework_df = get_config_file(bayes_op_framework, bayes_op_model)
    else:
        # Read bayesian configurations from file
        config_ini_name = ""
        config_excel = model_config["type_of_model"]["bayes_config_file"]
        get_config_data = mch.Config(config_ini_name)
        model_df, framework_df = get_config_data.get_config(config_excel)

    print("---Bayesian Framework config---")
    print(framework_df)
    print("---Bayesian model config---")
    print(model_df)

    # Saving the Bayesian configurations to file:
    d2 = datetime.now()
    d2 = d2.strftime("%m%d%Y_%H%M%S")
    wb = (
        wb
    ) = f'{out_path}/Bayesian_model_configurations_{global_config["run_name"]}_{d2}.xlsx'
    writer = pd.ExcelWriter(wb)
    framework_df.to_excel(writer, "Framework", index=False)
    model_df.to_excel(writer, "Model", index=False)
    writer.save()

    coef_df, y_train_pred, y_test_pred, y_pred = build_bayesian(
        X_train,
        y_train,
        X_test,
        y_test,
        X,
        y,
        model_config,
        global_config,
        framework_df,
        model_df,
    )
    # beta_df = coef_df.copy()
    coef_df = coef_df.iloc[:, 0:2]
    coef_df.columns = ["column", "beta"]
    coef_df2 = coef_df.copy()

    # Saving Bayesian output
    coeff_op_file = spliter(
        coef_df,
        bayesian_config["model"]["grouping_var"],
        global_config,
        out_path,
    )
    print(coeff_op_file)
    bayes_op_file = merger(
        coeff_op_file, model_df, framework_df, global_config, out_path
    )

    print("-----Bayesian output file name-----")
    print(bayes_op_file)

    return (
        coef_df2,
        coef_df,
        y_train_pred,
        y_test_pred,
        y_pred,
        y_train,
        y_test,
        y,
        X_train,
        X_test,
        X,
    )


def create_charts(
    data,
    X_train_o,
    y_train_o,
    y_train_pred,
    X_test_o,
    y_test_o,
    y_test_pred,
    global_config,
    model_config,
    coef_df,
    metrics,
    out_path="",
):
    d2 = datetime.now()
    d2 = d2.strftime("%m%d%Y_%H%M%S")

    report_dict = {}
    wb = f'{out_path}/model_output_{global_config["run_name"]}_{d2}_{model_config["type_of_model"]["algo"]}.xlsx'
    writer = pd.ExcelWriter(wb)
    train = X_train_o
    train[model_config["dv"]] = y_train_o
    train["y_pred"] = np.exp(y_train_pred)
    test = X_test_o
    test[model_config["dv"]] = y_test_o
    test["y_pred"] = np.exp(y_test_pred)
    data.to_excel(writer, "full_data", index=False)
    train.to_excel(writer, "train_data", index=False)
    test.to_excel(writer, "test_data", index=False)
    tc = model_config["time_column"]
    dv = model_config["dv"]
    train_plot = (
        hvPlot(train)
        .line(x=tc, y=["y_pred", dv], legend="top", height=500, width=950)
        .opts(legend_position="top_left", xrotation=90)
    )
    test_plot = (
        hvPlot(test)
        .line(x=tc, y=["y_pred", dv], legend="top", height=500, width=950)
        .opts(legend_position="top_left", xrotation=90)
    )
    full_plot = (
        hvPlot(data)
        .line(x=tc, y=["y_pred", dv], legend="top", height=500, width=950)
        .opts(legend_position="top_left", xrotation=90)
    )
    report_dict["Actual_vs_Predicted"] = {}
    report_dict["Actual_vs_Predicted"]["total_data"] = {}
    report_dict["Actual_vs_Predicted"]["total_data"]["plot"] = full_plot
    report_dict["Actual_vs_Predicted"]["train_data"] = {}
    report_dict["Actual_vs_Predicted"]["train_data"]["plot"] = train_plot
    report_dict["Actual_vs_Predicted"]["test_data"] = {}
    report_dict["Actual_vs_Predicted"]["test_data"]["plot"] = test_plot
    if model_config["type_of_model"]["algo"] == "lasso":
        coef_df.to_excel(writer, "coeficients", index=False)
        coef_df = coef_df.reset_index(drop=True)
        report_dict["coeficients"] = coef_df
    elif model_config["type_of_model"]["algo"] == "bayesian":
        coef_df.to_excel(writer, "coeficients", index=False)
        coef_df = coef_df.reset_index(drop=True)
        report_dict["coeficients"] = coef_df
    metrics.to_excel(writer, "metrics", index=False)
    report_dict["model_metrics"] = metrics
    # final_report_dict = {"Model Output": report_dict}
    create_report(
        report_dict,
        name=f'model_output_{global_config["run_name"]}_{d2}_{model_config["type_of_model"]["algo"]}',
        path=out_path,
        format=".html",
        split_sheets=True,
        tiger_template=False,
    )
    workbook = writer.book
    sheets = ["full_data", "train_data", "test_data"]
    for i in sheets:
        if i == "full_data":
            cols = data.columns.tolist()
            last_row = data.shape[0]
            cat_first_col = cols.index(model_config["time_column"])
            val1_first_col = cols.index(model_config["dv"])
            val2_first_col = cols.index("y_pred")
        elif i == "train_data":
            cols = train.columns.tolist()
            last_row = train.shape[0]
            cat_first_col = cols.index(model_config["time_column"])
            val1_first_col = cols.index(model_config["dv"])
            val2_first_col = cols.index("y_pred")
        else:
            cols = test.columns.tolist()
            last_row = test.shape[0]
            cat_first_col = cols.index(model_config["time_column"])
            val1_first_col = cols.index(model_config["dv"])
            val2_first_col = cols.index("y_pred")
        worksheet = writer.sheets[i]
        chart = workbook.add_chart({"type": "line"})
        chart.set_size({"width": 720, "height": 300})
        chart.add_series(
            {
                "name": [i, 0, val1_first_col],
                "categories": [i, 1, cat_first_col, last_row, cat_first_col],
                "values": [i, 1, val1_first_col, last_row, val1_first_col],
            }
        )
        chart.add_series(
            {
                "name": [i, 0, val2_first_col],
                "categories": [i, 1, cat_first_col, last_row, cat_first_col],
                "values": [i, 1, val2_first_col, last_row, val2_first_col],
            }
        )

        worksheet.insert_chart("E5", chart)
    writer.save()
    print("saved workbook " + wb)
    return data, train, test


def get_model_results(
    data: pd.DataFrame,
    model_config: dict,
    global_config: dict,
    bayesian_config: dict,
    out_path="",
):
    """Read model config and builds model and gets the results.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset used to build the model
    model_config: dict
        Model_config from config.yml
    global_config: dict
        Global_config from config.yml
    bayesian_config: dict
        Bayesian config from config.yml
    out_path: str
        Output path to save model results

    Returns
    -------
    data: pd.DataFrame
        Modelled data
    coef_df: pd.DataFrame
        Model coefficients
    y_pred: np.ndarray
        Model predictions
    """
    # import hvplot.pandas
    try:
        d2 = datetime.now()
        d2 = d2.strftime("%m%d%Y_%H%M%S")

        step = 0
        data = subset_columns(data, model_config)
        data = data.fillna(0)

        step = 1
        X_train, X_test, y_train, y_test = trn_test_split(data, model_config)
        X_train_o, X_test_o, y_train_o, y_test_o = (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            y_test.copy(),
        )
        if model_config["sales_dollars_column"] is not None:
            X = data.drop(
                [model_config["dv"], model_config["sales_dollars_column"]], axis=1
            )
        else:
            X = data.drop([model_config["dv"]], axis=1)
        y = data[model_config["dv"]]
        X_train.drop(model_config["time_column"], axis=1, inplace=True)
        X_test.drop(model_config["time_column"], axis=1, inplace=True)

        step = 2
        X_train, X_test, X = scale_variables(X_train, X_test, X, model_config)

        step = 3
        if model_config["type_of_model"]["type"] == "multiplicative":
            if model_config["type_of_model"]["transformation"] == "semilog":
                y_train, y_test, y = semi_log_transformation(y_train, y_test, y)
            else:
                X_train, X_test, X, y_train, y_test, y = log_transformation(
                    X_train, X_test, X, y_train, y_test, y, model_config
                )
        X_train = X_train.fillna(method="bfill")
        X_test = X_test.fillna(method="bfill")
        X = X.fillna(method="bfill")

        step = 4
        if model_config["type_of_model"]["algo"] == "lasso":
            if len(model_config["columns"]["categorical_vars"]) > 0:
                Xtrain = X_train.select_dtypes(exclude=["object"])
                Xtest = X_test.select_dtypes(exclude=["object"])
                XX = X.select_dtypes(exclude=["object"])
            else:
                Xtrain = X_train.copy()
                Xtest = X_test.copy()
                XX = X.copy()
            coef_df, y_train_pred, y_test_pred, y_pred = build_lasso(
                Xtrain, y_train, Xtest, y_test, XX, y, model_config
            )
            wb = f'{out_path}/model_output_{global_config["run_name"]}_{d2}_{model_config["type_of_model"]["algo"]}.xlsx'
            writer = pd.ExcelWriter(wb)
            coef_df2 = coef_df.copy()
            coef_df.to_excel(writer, "Lasso coefficients", index=False)
            pd.DataFrame(y_pred).to_excel(writer, "Lasso Predictions", index=False)
            writer.save()
            # pd.to_csv("Lasso_coef_df", coef_df)

        step = 5
        if model_config["type_of_model"]["algo"] == "bayesian":
            (
                coef_df2,
                coef_df,
                y_train_pred,
                y_test_pred,
                y_pred,
                y_train,
                y_test,
                y,
                X_train,
                X_test,
                X,
            ) = bayesian_model_func(
                data,
                X_train,
                X_test,
                X,
                y_train,
                y_test,
                y,
                model_config,
                bayesian_config,
                global_config,
                out_path="",
            )
        step = 6
        metrics = get_metrics(
            y_train,
            y_train_pred,
            y_test,
            y_test_pred,
            y,
            y_pred,
            X_train,
            X_test,
            X,
            model_config,
        )
        for c in X.columns.tolist():
            data[c] = X[c]
        data[model_config["dv"]] = y
        data["y_pred"] = y_pred
        data = get_exp_df(data, model_config)
        step = 7
        data, train, test = create_charts(
            data,
            X_train_o,
            y_train_o,
            y_train_pred,
            X_test_o,
            y_test_o,
            y_test_pred,
            global_config,
            model_config,
            coef_df,
            metrics,
            out_path,
        )
    except Exception:
        dict_err_stage = {
            0: "subset columns",
            1: "split train and test data",
            2: "variable scaling",
            3: "log transformation",
            4: "build Lasso",
            5: "build Bayesian",
            6: "get model metrics",
            7: "all good",
        }
        logging.error(f"Exception occurred while {dict_err_stage[step]}", exc_info=True)
    return data, coef_df2, y_pred
