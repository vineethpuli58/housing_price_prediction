# isort: skip_file
# import hvplot.pandas
import logging
import math
from datetime import datetime

import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models.formatters import DatetimeTickFormatter
from hvplot import hvPlot
from statsmodels.tsa.filters.filtertools import recursive_filter

from tigerml.core.reports import create_report

hv.extension("bokeh")


def impute_missing_values(data: pd.DataFrame):
    """Impute missing values.

    Parameters
    ----------
    data : pd.DataFrame
        Input Dataset

    Returns
    -------
    pd.DataFrame
        Missing value imputed dataset
    """
    logging.info("Entering impute_missing_vlaues function")
    m_df = pd.DataFrame(data.isna().sum())
    m_df = m_df.reset_index()
    m_df.columns = ["column_name", "count"]
    m_df = m_df.loc[m_df["count"] > 0, :]
    for c in m_df["column_name"].tolist():
        data[c] = data[c].fillna(method="bfill")
        if data[c].isna().sum() > 0:
            data[c] = data[c].fillna(method="ffill")
    logging.info("Leaving impute_missing_vlaues function")
    return data


def adstock_decay(x: float):
    """Calculate the decay rate of the halflife.

    Parameters
    ----------
    x : float
        Halflife value

    Returns
    -------
    float
        Decay rate of half life
    """
    return math.exp(math.log(0.5) / x)


def get_best_transformation(corr: pd.DataFrame, dv: str):
    """Find the best adstock or stocking lag of the variable based on correlation with dv.

    Parameters
    ----------
    corr: pd.DataFrame
        Correlation matrix
    dv: str
        Dependent variable in dataset

    Returns
    -------
    str
        Best adstock/stocking_lag variable
    """
    corr = corr.loc[corr["index"] != dv, :]
    corr = corr.loc[corr[dv] == corr[dv].max(), :]
    return corr.iloc[0, :]["index"]


def get_adstock(col: pd.Series, halflife: int):
    """Adstock calculation method.

    Parameters
    ----------
    col: pd.Series
        Column from feature_config to compute adstock.
    halflife: int
        Corresponding halflife of the column given in the feature_config.

    Returns
    -------
    pd.Series
        Adstock value of the column for particular halflife
    """
    ad = adstock_decay(halflife)
    ad_stock = recursive_filter(col, ad)
    return ad_stock


def create_adstock(data: pd.DataFrame, ad_stock_vars: list, feature_config: dict):
    """Create adstocks for the list of variables with different decay rates.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    ad_stock_vars: list
        list of independent variables for which adstock needs to be computed
    feature_config: dict
        feature config from config.yml

    Returns
    -------
    pd.DataFrame
        Dataset with adstock transformations
    """
    ad_stock_df = pd.DataFrame()
    tmp_df = pd.DataFrame()

    gv = feature_config["gv"]
    for i in list(data[gv].unique()):
        if tmp_df.shape[0] == 0:
            tmp_df = data[(data[gv] == i)]
        else:
            ad_stock_df = pd.concat([ad_stock_df, tmp_df])
            tmp_df = data[(data[gv] == i)]
        for j in ad_stock_vars:
            adstock_range_list = feature_config["adstocks"][j]
            adstock_range_list = adstock_range_list.split(",")
            adstock_range_list = [int(x) for x in adstock_range_list]
            for k in adstock_range_list:
                tmp_df[j + "_" + str(k) + "_adstock"] = get_adstock(tmp_df[j], k)

    ad_stock_df = pd.concat([ad_stock_df, tmp_df])
    return ad_stock_df


def get_adstock_data(
    base_df: pd.DataFrame, adstock_columns: list, feature_config: dict
):
    """Read the data and apply adstock transformation for the given variables.

    Parameters
    ----------
    base_df: pd.DataFrame
        Input Dataset
    adstock_columns: list
        list of columns for which transformation has to be applied
    feature_config: dict
        feature config from config.yml

    Returns
    -------
    pd.DataFrame
        Dataset with adstock transformations
    """
    adstock_df = create_adstock(base_df, adstock_columns, feature_config)
    return adstock_df


def adstock_analysis(
    data: pd.DataFrame, feature_config: dict, dv: str, global_config: dict
):
    """Adstock analysis based on config.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    feature_config: dict
        Feature_config from config.yml
    dv: str
        Dependent variable given in global_config
    global_config: dict
        Global_config from config.yml

    Returns
    -------
    data : pd.DataFrame
        Dataset with adstock columns
    output_df : pd.DataFrame
        Dataframe with added adstock transformations
    """
    logging.info("Entering adstocks analysis function")
    adstock_cols = list(feature_config["adstocks"].keys())
    output_df = pd.DataFrame()

    ad_stock_df = get_adstock_data(data, adstock_cols, feature_config)

    data_adstock = ad_stock_df.copy()
    cols = (
        [feature_config["gv"]]
        + [global_config["dv"]]
        + adstock_cols
        + list(ad_stock_df.columns[ad_stock_df.columns.str.endswith("_adstock")])
    )

    for i in ad_stock_df.columns:
        if i not in cols:
            ad_stock_df.drop(i, inplace=True, axis="columns")

    for i in list(ad_stock_df[feature_config["gv"]].unique()):
        ad_df = ad_stock_df[(ad_stock_df[feature_config["gv"]] == i)]
        for a in adstock_cols:
            adstock_df = ad_df[ad_df.columns[ad_df.columns.str.contains(a)]]
            adstock_df[a] = ad_df[a]
            hl = feature_config["adstocks"][a]
            hl = hl.split(",")
            hl = [int(x) for x in hl]
            adstock_df[dv] = ad_df[dv]

            corr = adstock_df.corr().reset_index()
            corr = corr.loc[:, ["index", dv]]
            dummy_df = pd.DataFrame()
            columns = ["column"]
            dummy_df["column"] = [a]
            for h in hl:
                columns.append(f"{h}_adstock")
            for c in columns:
                if c != "column":
                    dummy_df[c] = corr.loc[corr["index"] == f"{a}_{c}", :][dv].tolist()
            dummy_df[feature_config["gv"]] = [i]
            output_df = output_df.append(dummy_df)
            # Selecting best transformation
            if feature_config["select_best_transformation"]:
                fn = get_best_transformation(corr, dv)
                data[fn] = adstock_df[fn]
            else:
                # data = pd.concat([data, adstock_df.drop([dv, a], axis=1)], axis=1)
                data = data_adstock
            output_df["transformation"] = "adstock"
    logging.info("Leaving adstocks analysis function")
    return data, output_df


def s_curve_prep_params(data: pd.DataFrame, feature_config: dict):
    """Prepare S curve transformation params.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset on which the transformation is to be applied
    feature_config: dict
        Feature config from config.yml

    Returns
    -------
    s_vars: list
        Variables on which transformation is to be applied
    alpha: list
        List of max peak values for each series
    beta: list
        List of values to control the pattern of each variable
    """
    # Gets all the variables given in the config and the adstock variables
    s_vars = feature_config["s_curve_vars"] + list(
        data.columns[data.columns.str.endswith("_adstock")]
    )
    alpha = list()
    # Beta values control the pattern that are data specific
    beta = [
        0.00000064,
        0.000012,
        0.0000055,
        0.0000058,
        0.00000065,
        0.00000055,
        0.0000045,
        0.0000075,
        0.00000082,
        0.00000085,
    ]
    # Alpha values are the max peak of series that are data specific
    for i in s_vars:
        a = round(np.percentile(data[i], 90), 2)
        alpha.append(a)
    return (s_vars, alpha, beta)


def Mathematical_S_curve(alpha: float, beta: float, value: float):
    """Calculate S curve value.

    Parameters
    ----------
    alpha: float
        Alpha parameter
    beta: float
        Beta parameter
    value: float
        Value

    Returns
    -------
    float
        Computed S curve value
    """
    return alpha * (1 - math.exp(-1 * beta * value))


def Mathematical_S_curve2(beta: float, value: float):
    """Calculate S curve value.

    Parameters
    ----------
    beta: beta parameter
    value: value

    Returns
    -------
    S transformed float value
    """
    if value > 0:
        return 1 / (1 + math.exp(-1 * beta * value))
    else:
        return value


def S_curve_to_column(
    alpha: float, beta: float, dataframe: pd.DataFrame, column_name: str
):
    """Apply scurve to a pandas series.

    Parameters
    ----------
    alpha: float
        Alpha value
    beta: float
        Beta value
    dataframe: pd.DataFrame
        Dataset to compute S curve transformation
    column_name:  str
        Column to apply S curve transformation

    Returns
    -------
    pd.DataFrame
        Dataset with scurve transformations
    """
    new_col_name = f"S_{column_name}"
    dataframe[new_col_name] = dataframe[column_name].apply(
        lambda val: Mathematical_S_curve(alpha, beta, val)
    )
    return dataframe


def get_s_transform_dict(s_param: list, alpha: list, beta: list):
    """Prepare S curve parameter dictionary.

    Parameters
    ----------
    s_param: list
        List of scurve parameters
    alpha: list
        List of alpha values
    beta: list
        List of beta values

    Returns
    -------
    dict
        Dictionary with variable and alpha,beta parameters
    """
    s_transform_dict = dict()
    s_transform_dict = {key: value for key, *value in zip(s_param, alpha, beta)}
    return s_transform_dict


def S_curve_transformation(dataframe: pd.DataFrame, S_transform_dict: dict):
    """Apply s curve transformation for a set of varibales.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataset to which Scurve needs to be applied
    S_transform_dict: dict
        Dictionary with column and aplha,beta values

    Returns
    -------
    pd.DataFrame
        Dataset with scurve transformations
    """
    logging.info("Entering S curve transformation function")
    df_transformed = dataframe.copy()
    for key, value in S_transform_dict.items():
        df_transformed = S_curve_to_column(value[0], value[1], df_transformed, key)
    logging.info("Leaving S curve transformation function")
    return df_transformed


def get_S_transformation_corr(
    data: pd.DataFrame, global_config: dict, feature_config: dict
):
    """Intermediate function to get S curve transformation.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataset
    global_config: dict
        Global config from config.yml
    feature_config: dict
        Feature config from config.yml

    Returns
    -------
    data: pd.DataFrame
        Dataset with S curve transformations
    output_df: pd.DataFrame
        Dataframe with best transformations
    """
    s_data = data.copy()
    dv = global_config["dv"]
    output_df = pd.DataFrame()
    for i in list(s_data[feature_config["gv"]].unique()):
        s_df = s_data[(s_data[feature_config["gv"]] == i)]
        for a in feature_config["s_curve_vars"]:
            # print(a)
            s_curve_df = s_df[s_df.columns[s_df.columns.str.contains(a)]]
            s_curve_df[a] = s_df[a]
            s_curve_df[dv] = s_df[dv]
            corr = s_curve_df.corr().reset_index()
            corr = corr.loc[:, ["index", dv]]
            dummy_df = pd.DataFrame()
            columns = ["column"]
            dummy_df["column"] = [a]
            columns.append("coefficient")
            for c in columns:
                if c != "column":
                    dummy_df[c] = corr.loc[corr["index"] == f"{a}", :][dv].tolist()
            dummy_df[feature_config["gv"]] = [i]
            output_df = output_df.append(dummy_df)
            # Selecting best transformation
            if feature_config["select_best_transformation"]:
                fn = get_best_transformation(corr, dv)
                data[fn] = s_curve_df[fn]
            else:
                # data = pd.concat([data, adstock_df.drop([dv, a], axis=1)], axis=1)
                data = s_data
            output_df["transformation"] = "S_curve"
            output_df["parameter"] = ""
            cols_seq = [
                "column",
                "transformation",
                feature_config["gv"],
                "parameter",
                "coefficient",
            ]
            output_df = output_df[cols_seq]
    return data, output_df


def stocking_lag(col: str, lag: int):
    """Return stocking lag of the variable takes lag as input.

    Parameters
    ----------
    col: str
        Column from feature_config stocking_lags value
    lag: int
        Corresponding lag of the column from feature_config

    Returns
    -------
    float
        Stocking_lag value of the column.
    """
    # ads_decay = 1
    adstock_decays = [1 for i in range(1, lag)]
    adf = pd.DataFrame()
    adf["x"] = col
    for i in range(len(adstock_decays)):
        adf[f"lag_{i+1}"] = adf["x"].shift(periods=i + 1)
        adf[f"lag_{i+1}"] = adf[f"lag_{i+1}"] * adstock_decays[i]
        adf[f"lag_{i+1}"] = adf[f"lag_{i+1}"].fillna(0)
    adf["adstock"] = adf["x"]
    for i in range(len(adstock_decays)):
        adf["adstock"] = adf["adstock"] + adf[f"lag_{i+1}"]
    return adf["adstock"]


def create_lag(
    dataframe: pd.DataFrame, column_name: str, lag_no: int, feature_config: dict
):
    """Create lag effect on varibales at geography level.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataset on which lag has to be applied
    column_name: str
        Column name for which lag effect to be applied
    lag_no: int
        Duration of lag effect
    feature_config: dict
        feature config from config.yml

    Returns
    -------
    pd.DataFrame
        Dataset with lag_effect included
    """
    new_col_name = f"{column_name}_lag_{lag_no}"
    gv = feature_config["gv"]
    lag_data = dataframe.groupby(gv)[column_name].shift(lag_no)
    lag_data = pd.DataFrame({new_col_name: lag_data})
    lag_data.fillna(0, inplace=True)
    df_with_lag = pd.merge(dataframe, lag_data, left_index=True, right_index=True)
    return df_with_lag


def create_lag_effect(dataframe: pd.DataFrame, lag_columns: list, feature_config: dict):
    """Apply lag effect on a list of columns.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataset on which lag has to be applied
    lag_columns: list
        List of variables with duration of lag-effect for which lag effect to be applied eg:[['col',2]]
    feature_config: dict
        feature config from config.yml

    Returns
    -------
    pd.DataFrame
        Dataset with lag_effect included
    """
    temp_df = dataframe.copy()
    for lag_column in lag_columns:
        temp_df = create_lag(temp_df, lag_column[0], lag_column[1], feature_config)
    return temp_df


def s_lags_analysis(
    data: pd.DataFrame, feature_config: dict, dv: str, global_config: dict
):
    """Adstock analysis based on config.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    feature_config: dict
        Feature_config from config.yml
    dv: str
        Dependent variable given in global_config
    global_config: dict
        Global_config from config.yml

    Returns
    -------
    data : pd.DataFrame
        Dataset with s-lag variables
    output_df : pd.DataFrame
        Dataframe with added s-lag transformations
    """
    logging.info("Entering stocking lags analysis function")
    # Selecting columns for Stocking lag
    sl_cols = list(feature_config["stocking_lags"].keys())
    output_df = pd.DataFrame()
    lag_columns = []
    for a in sl_cols:
        sl_df = pd.DataFrame()
        sl_df[a] = data[a]
        lags = feature_config["stocking_lags"][a]
        lags = lags.split(",")
        lags = [int(x) for x in lags]
        for lag in lags:
            lag_columns.append([a, lag])
    # Creating lag effect
    data_lag = create_lag_effect(data, lag_columns, feature_config)

    for i in list(data_lag[feature_config["gv"]].unique()):
        lag_df = data_lag[(data_lag[feature_config["gv"]] == i)]
        for a in sl_cols:
            stock_lag_df = lag_df[lag_df.columns[lag_df.columns.str.contains(a)]]
            stock_lag_df[a] = lag_df[a]
            lags = feature_config["stocking_lags"][a]
            lags = lags.split(",")
            lags = [int(x) for x in lags]
            stock_lag_df[dv] = lag_df[dv]

            corr = stock_lag_df.corr().reset_index()
            corr = corr.loc[:, ["index", dv]]
            dummy_df = pd.DataFrame()
            columns = ["column"]
            dummy_df["column"] = [a]
            for lag in lags:
                columns.append(f"lag_{lag}")
            for c in columns:
                if c != "column":
                    dummy_df[c] = corr.loc[corr["index"] == f"{a}_{c}", :][dv].tolist()
            dummy_df[feature_config["gv"]] = [i]
            output_df = output_df.append(dummy_df)
            # Selecting best transformation
            if feature_config["select_best_transformation"]:
                fn = get_best_transformation(corr, dv)
                data[fn] = stock_lag_df[fn]
            else:
                # data = pd.concat([data, adstock_df.drop([dv, a], axis=1)], axis=1)
                data = data_lag
            output_df["transformation"] = "stocking_lag"
    logging.info("Leaving stocking lags analysis function")
    return data, output_df


def get_transformation_cols(feature_config: dict):
    """Intermediate function to get transformation columns.

    Parameters
    ----------
    feature_config: dict
        Feature config from config.yml

    Returns
    -------
    cols: list
        List of transformation columns
    """
    if feature_config["stocking_lags"] is None:
        cols = list(feature_config["adstocks"].keys())
    elif feature_config["adstocks"] is None:
        cols = list(feature_config["stocking_lags"].keys())
    else:
        cols = list(feature_config["adstocks"].keys()) + list(
            feature_config["stocking_lags"].keys()
        )
    return cols


def tranformations_bivariate_plots(
    data: pd.DataFrame, feature_config: dict, global_config: dict, out_path=""
):
    """Create timeseries plots for all the transformed variables.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset with transformations
    feature_config: dict
        Feature_config from config.yml
    global_config: dict
        Global_config from config.yml
    out_path: str
        Output path to save Bivariate plots
    """
    d2 = datetime.now()
    d2 = d2.strftime("%m%d%Y_%H%M%S")
    cols = get_transformation_cols(feature_config)
    cols = list(set(cols))
    feature_dict = {}
    for c in cols:
        feature_dict[c] = {}
        if feature_config["adstocks"] is not None:
            if c in list(feature_config["adstocks"].keys()):
                a = feature_config["adstocks"][c].split(",")
                feature_dict[c]["adstocks"] = [int(x) for x in a]
        if feature_config["stocking_lags"] is not None:
            if c in list(feature_config["stocking_lags"].keys()):
                a = feature_config["stocking_lags"][c].split(",")
                feature_dict[c]["stocking_lags"] = [int(x) for x in a]
    report_dict = {"Transformations Bivariate Plots": {}}
    for c in cols:
        t_df = pd.DataFrame()
        t_df[global_config["time_column"]] = data[global_config["time_column"]].astype(
            "datetime64[ns]"
        )
        t_df[c] = data[c]
        t_df[global_config["dv"]] = data[global_config["dv"]]
        fd_keys = feature_dict[c].keys()
        for f in list(fd_keys):
            if f == "adstocks":
                halflifes = feature_dict[c]["adstocks"]
                for h in halflifes:
                    t_df[f"halflife_{h}"] = get_adstock(data[c], h)
            if f == "stocking_lags":
                sls = feature_dict[c]["stocking_lags"]
                for s in sls:
                    t_df[f"stocking_lag_{s}"] = stocking_lag(data[c], s)
        ycols = [x for x in t_df.columns.tolist() if x != global_config["time_column"]]
        tc = global_config["time_column"]
        # dv = global_config["dv"]
        t_df = t_df.set_index(tc)
        plot = (
            hvPlot(t_df)
            .line(
                x=tc,
                y=ycols,
                legend="top",
                height=600,
                width=1000,
                by=["index.year", "index.month"],
            )
            .opts(
                legend_position="top",
                xrotation=90,
                xformatter=DatetimeTickFormatter(months="%b %Y"),
            )
        )
        cor = t_df.corr().reset_index()
        cor = cor.loc[
            cor["index"] != global_config["dv"], ["index", global_config["dv"]]
        ]
        cor.columns = ["tranformation", "correlation"]
        cor = cor.reset_index(drop=True)
        report_dict["Transformations Bivariate Plots"][c] = {
            "plot": plot,
            "correlation": cor,
        }
    create_report(
        report_dict,
        name=f'tranformation_bivariate_plots_{global_config["run_name"]}_{d2}',
        path=out_path,
        format=".html",
        split_sheets=True,
        tiger_template=False,
    )


def get_required_format(df: pd.DataFrame, feature_config: dict):
    """Intermediate function to get transformations in required format.

    Parameters
    ----------
    df: pd.DataFrame
    feature_config: dict

    Returns
    -------
    pd.DataFrame
        Dataframe with new transformations
    """
    gv = feature_config["gv"]
    if gv + "_x" in df.columns:
        df[gv] = df[gv + "_x"].combine_first(df[gv + "_y"])
        df.drop([gv + "_x", gv + "_y"], axis="columns", inplace=True)
    keys = [c for c in df.columns if "_" in c]
    df = pd.melt(
        df,
        id_vars=["column", "transformation", gv],
        value_vars=keys,
        value_name="coefficient",
    )
    df.dropna(inplace=True)
    df.rename(columns={"variable": "parameter"}, inplace=True)
    return df


def feature_eng(
    data: pd.DataFrame, global_config: dict, feature_config: dict, out_path=""
):
    """Feature Engineering execution wrapper.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    global_config: dict
        Global_config parameters from config.yml
    feature_config: dict
        Feature_config parameters from config.yml
    out_path:
        Output path where feature engineering outputs are saved

    Returns
    -------
    data: pd.DataFrame
        Feature transformed dataset
    final_df: pd.DataFrame
        Dataset with feature transformations
    """
    d2 = datetime.now()
    d2 = d2.strftime("%m%d%Y_%H%M%S")
    try:
        data = impute_missing_values(data)
    except Exception:
        logging.error("Exception occurred while imputing missing values", exc_info=True)
    try:
        step = 0
        if (
            feature_config["adstocks"] is not None
            and len(feature_config["adstocks"].keys()) > 0
        ):
            # if len(feature_config["adstocks"].keys()) > 0:
            data, ads_df = adstock_analysis(
                data, feature_config, global_config["dv"], global_config
            )
        else:
            ads_df = pd.DataFrame()
        step = 1
        if len(feature_config["s_curve_vars"]) > 0:
            data_s_transformed = pd.DataFrame()
            for i in list(data[feature_config["gv"]].unique()):
                data_s = data[(data[feature_config["gv"]] == i)]
                s_vars, alpha, beta = s_curve_prep_params(data_s, feature_config)
                s_transform_dict = get_s_transform_dict(s_vars, alpha, beta)
                data_geo = S_curve_transformation(data_s, s_transform_dict)
                data_s_transformed = pd.concat([data_s_transformed, data_geo], axis=0)
            data = data_s_transformed.copy()
            data, s_df = get_S_transformation_corr(data, global_config, feature_config)
        step = 2
        if (
            feature_config["stocking_lags"] is not None
            and len(feature_config["stocking_lags"].keys()) > 0
        ):
            # if len(feature_config["stocking_lags"].keys()) > 0:
            data, sl_df = s_lags_analysis(
                data, feature_config, global_config["dv"], global_config
            )
        else:
            sl_df = pd.DataFrame()
        step = 3
        ads_df.columns = [
            x.replace("_adstock", "halflife_adstock") if len(x) == 2 else x
            for x in ads_df.columns.tolist()
        ]
        sl_df.columns = [
            x.replace("lag_", "stocking_lag_") if len(x) == 3 else x
            for x in sl_df.columns.tolist()
        ]
        if ads_df.empty:
            final_df = sl_df
        elif sl_df.empty:
            final_df = ads_df
        else:
            final_df = ads_df.merge(sl_df, on="column", how="outer")
            final_df["transformation"] = np.where(
                final_df["transformation_x"].isnull(),
                final_df["transformation_y"],
                final_df["transformation_x"],
            )
            final_df.drop(
                ["transformation_x", "transformation_y"], axis=1, inplace=True
            )
        cl = final_df.columns.tolist()
        clss = ["column", "transformation"] + [
            x for x in cl if x not in ["column", "transformation"]
        ]
        final_df = final_df.loc[:, clss]
        final_df = get_required_format(final_df, feature_config)
        final_df = pd.concat([final_df, s_df], ignore_index=True)
        final_df.to_csv(
            f'{out_path}/correlation_transformations_{global_config["run_name"]}_{d2}.csv',
            index=False,
        )
        step = 4
        tranformations_bivariate_plots(data, feature_config, global_config, out_path)
        step = 5
    except Exception:
        dict_err_stage = {
            0: "doing adstock analysis",
            1: "applying s curve transformations",
            2: "doing stocking lags analysis",
            3: "saving adstock and stockling lag analysis",
            4: "getting transformations time series plots",
            5: "all good",
        }
        logging.error(f"Exception occurred while {dict_err_stage[step]}", exc_info=True)

    return data, final_df
