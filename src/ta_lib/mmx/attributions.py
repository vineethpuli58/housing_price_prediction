# isort: skip_file
# import hvplot.pandas
import logging
import math
from datetime import datetime

import holoviews as hv
import numpy as np
import pandas as pd
from hvplot import hvPlot

from tigerml.core.reports import create_report

hv.extension("bokeh")


def get_var_contribution_variants(
    dist_df: pd.DataFrame,
    var_col_name: str,
    value_col_name: str,
    group: str,
    attribution_config: dict,
):
    """Get variable contribution by different variants.

    Parameters
    ----------
    dist_df: pd.DataFrame
        Dataframe consisting of IDVs
    var_col_name: str
        Column name for melted IDV columns
    value_col_name: str
        Column name for IDV values
    group: str
        Unique value of grouping variable
    attribution_config: dict
        Attribution config from config.yml

    Returns
    -------
    dist_df_1: pd.DataFrame
        Datewise contribution data
    dist_df_2: pd.DataFrame
        Quarterly contribution data
    dist_df_3: pd.DataFrame
        Yearly contribution data
    """
    gv = attribution_config["gv"]
    # Datewise Aggregation
    pct_dist_df = dist_df.copy()
    numeric_cols = pct_dist_df.select_dtypes(include="number").columns.to_list()
    pct_dist_df[numeric_cols] = pct_dist_df[numeric_cols].div(
        pct_dist_df["Predicted_sales_" + group], axis=0
    )

    dist_df_1 = pd.merge(
        dist_df.melt(
            id_vars=[attribution_config["time_column"]],
            var_name=var_col_name,
            value_name=value_col_name,
        ),
        pct_dist_df.melt(
            id_vars=[attribution_config["time_column"]],
            var_name=var_col_name,
            value_name="pct_" + value_col_name,
        ),
        how="left",
        on=[attribution_config["time_column"], var_col_name],
    )
    dist_df_1[gv] = group

    # Quarterly Aggegration
    qtr_dist_df = dist_df.copy()
    qtr_dist_df["Quarter"] = (
        qtr_dist_df[attribution_config["time_column"]].dt.year.astype(str)
        + "-"
        + "Q"
        + qtr_dist_df[attribution_config["time_column"]].dt.quarter.astype(str)
    )
    qtr_dist_df = qtr_dist_df.drop(columns=[attribution_config["time_column"]])
    qtr_dist_df = qtr_dist_df.groupby(by=["Quarter"], as_index=False).agg(np.sum)

    pct_qtr_dist_df = qtr_dist_df.copy()
    pct_qtr_dist_df[numeric_cols] = pct_qtr_dist_df[numeric_cols].div(
        pct_qtr_dist_df["Predicted_sales_" + group], axis=0
    )

    dist_df_2 = pd.merge(
        qtr_dist_df.melt(
            id_vars=["Quarter"], var_name=var_col_name, value_name=value_col_name
        ),
        pct_qtr_dist_df.melt(
            id_vars=["Quarter"],
            var_name=var_col_name,
            value_name="pct_" + value_col_name,
        ),
        how="left",
        on=["Quarter", var_col_name],
    )
    dist_df_2[gv] = group

    # Yearly Aggegration
    yr_dist_df = dist_df.copy()
    yr_dist_df["Year"] = yr_dist_df[attribution_config["time_column"]].dt.year.astype(
        str
    )
    yr_dist_df = yr_dist_df.drop(columns=[attribution_config["time_column"]])
    yr_dist_df = yr_dist_df.groupby(by=["Year"], as_index=False).agg(np.sum)

    pct_yr_dist_df = yr_dist_df.copy()
    pct_yr_dist_df[numeric_cols] = pct_yr_dist_df[numeric_cols].div(
        pct_yr_dist_df["Predicted_sales_" + group], axis=0
    )

    dist_df_3 = pd.merge(
        yr_dist_df.melt(
            id_vars=["Year"], var_name=var_col_name, value_name=value_col_name
        ),
        pct_yr_dist_df.melt(
            id_vars=["Year"], var_name=var_col_name, value_name="pct_" + value_col_name
        ),
        how="left",
        on=["Year", var_col_name],
    )
    dist_df_3[gv] = group
    return dist_df_1, dist_df_2, dist_df_3


def _predict(
    pred_df: pd.DataFrame,
    model_coef: pd.DataFrame,
    intercept_name: str,
    attribution_config: dict,
    var_col="model_coefficient_name",
    coef_col="model_coefficient_value",
):
    """Predict Sales.

    Parameters
    ----------
    pred_df: pd.DataFrame
        Dataframe consisting of IDVs
    model_coef: pd.DataFrame
        Dataframe consisting of model coefficient and its value
    intercept_name: str
        Name of intercept variable
    attribution_config: dict
        Attribution config from config.yml
    var_col: str
        Column containing model variable
    coef_col: str
        Column containing model variables and their coefficient estimate

    Returns
    -------
    np.ndarray
        Predicted sales
    """
    if attribution_config["model_type"] == "Mixed_effect":
        # Random var coefficients
        if attribution_config["columns"]["random_effect_cols"] is not None:
            rand_var_dict = {}
            for v in attribution_config["columns"]["random_effect_cols"]:
                var = {}
                for i in pred_df[attribution_config["gv"]].unique():
                    df = model_coef[model_coef["column"].str.contains(v) is True]
                    j = df.loc[df[var_col].str.contains(i), coef_col].to_list()[0]
                    var[i] = j
                rand_var_dict[v] = var
            # Random var df
            rand_var_df = pred_df.loc[
                :, attribution_config["columns"]["random_effect_cols"]
            ]
            rand_values = []
            for v in attribution_config["columns"]["random_effect_cols"]:
                for i in pred_df[attribution_config["gv"]].unique():
                    rand_values1 = rand_var_df.values.dot(rand_var_dict[v][i])
                    rand_values.append(rand_values1)
            # Excluding Random effect cols:
            for i in range(0, len(attribution_config["columns"]["random_effect_cols"])):
                s = str(attribution_config["columns"]["random_effect_cols"][i])
                model_coef = model_coef[model_coef["column"].str.contains(s) is False]
        # random Intercepts
        rand_intercept = []
        for i in model_coef.column:
            if "intercept_" in i:
                rand_intercept.append(i)
        # Random Intercept coefficients
        # rand_intercept_coef = model_coef.loc[model_coef[var_col].isin(rand_intercept), coef_col]

        # Independent var cols
        idv_cols = [col for col in model_coef[var_col] if col != intercept_name]
        idv_cols = [col for col in idv_cols if col not in rand_intercept]
        # idv_cols = [col for col in idv_cols if col not in rand_cols]
        # Independent var coefficients
        idv_coef = model_coef.loc[model_coef[var_col].isin(idv_cols), coef_col]
        idv_df = pred_df.loc[:, idv_cols]

        rand_int_dict = {}
        for i in pred_df[attribution_config["gv"]].unique():
            j = model_coef.loc[model_coef[var_col].str.contains(i), coef_col].to_list()[
                0
            ]
            rand_int_dict[i] = j
        intercept = model_coef.loc[
            ~model_coef[var_col].isin(idv_cols), coef_col
        ].to_list()[0]

        prediction = []
        for i in pred_df[attribution_config["gv"]].unique():
            # preds = idv_df.values.dot(idv_coef.values) + intercept + rand_int_dict[i] + rand_values[j]
            preds = idv_df.values.dot(idv_coef.values) + intercept + rand_int_dict[i]
            prediction.append(preds)
    else:
        idv_cols = [col for col in model_coef[var_col] if col != intercept_name]
        idv_coef = model_coef.loc[model_coef[var_col].isin(idv_cols), coef_col]
        idv_df = pred_df.loc[:, idv_cols]
        # x = model_coef.loc[~model_coef[var_col].isin(idv_cols), coef_col]
        intercept = model_coef.loc[
            ~model_coef[var_col].isin(idv_cols), coef_col
        ].to_list()[0]
        prediction = idv_df.values.dot(idv_coef.values) + intercept
    return prediction


def get_random_intercept(
    df, model_coef, var_col, coef_col, rand_intercept, rand_int_dict, attribution_config
):
    for i in range(0, len(attribution_config["columns"]["random_effect_cols"])):
        s = str(attribution_config["columns"]["random_effect_cols"][i])
        model_coef = model_coef[model_coef["column"].str.contains(s) is False]
    # random Intercepts

    for i in model_coef.column:
        if "intercept_" in i:
            rand_intercept.append(i)

    for i in df[attribution_config["gv"]].unique():
        j = model_coef.loc[model_coef[var_col].str.contains(i), coef_col].to_list()[0]
        rand_int_dict[i] = j

    return rand_int_dict, rand_intercept


def get_var_contribution_wo_baseline_defined(
    df: pd.DataFrame,
    model_coef: pd.DataFrame,
    wk_sold_price: pd.DataFrame,
    var_col: str,
    coef_col: str,
    intercept_name: str,
    base_var: list,
    attribution_config: dict,
    all_df=None,
):
    """Get variable contribution without baseline defined.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    wk_sold_price : pd.DataFrame
        Dataframe consisting of weekly sold price
    all_df : pd.DataFrame
        Dataframe consisting of IDVs of all PPGs
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables' coefficient estimate
    intercept_name : str
        Name of intercept variable
    base_var : list of str
        Base variables
    attribution_config : dict
        Attribution config from config.yml

    Returns
    -------
    overall_dt_dist_df: pd.DataFrame
        Overall datewise distribution
    overall_qtr_dist_df: pd.DataFrame
        Overall quarterly distribution
    overall_yr_dist_df: pd.DataFrame
        Overall yearly distribution
    """
    # Predict Sales
    model_df = df.copy()
    prediction = np.exp(
        _predict(
            model_df, model_coef, intercept_name, attribution_config, var_col, coef_col
        )
    )
    for i, j in zip(
        model_df[attribution_config["gv"]].unique(), range(0, len(prediction))
    ):
        i = "Predicted_sales_" + i
        if attribution_config["model_type"] == "Mixed_effect":
            model_df[i] = prediction[j]
        else:
            model_df[i] = prediction

    coef_Df = model_coef.copy()
    if attribution_config["model_type"] == "Mixed_effect":
        rand_intercept = []
        rand_int_dict = {}
        rand_int_dict, rand_intercept = get_random_intercept(
            df,
            model_coef,
            var_col,
            coef_col,
            rand_intercept,
            rand_int_dict,
            attribution_config,
        )

    # Get base and impact variables
    base_var = [intercept_name] + base_var
    impact_var = [i for i in model_coef[var_col] if i not in base_var]
    if attribution_config["columns"]["random_effect_cols"] is not None:
        # base_var = [intercept_name] + base_var
        impact_var = [i for i in model_coef[var_col] if i not in base_var]
        impact_var = impact_var + attribution_config["columns"]["random_effect_cols"]
    if attribution_config["model_type"] == "Mixed_effect":
        impact_var = [i for i in impact_var if i not in rand_intercept]

    model_df[intercept_name] = 1
    gv = attribution_config["gv"]

    dt_unit_dist_df = pd.DataFrame()
    qtr_unit_dist_df = pd.DataFrame()
    yr_unit_dist_df = pd.DataFrame()
    dt_price_dist_df = pd.DataFrame()
    qtr_price_dist_df = pd.DataFrame()
    yr_price_dist_df = pd.DataFrame()

    for i in model_df[gv].unique():
        tmp_model_coef = model_coef[model_coef[var_col].isin(base_var)]
        tmp_model_df = model_df.copy()
        base_val = tmp_model_df[tmp_model_coef[var_col].to_list()].values.dot(
            tmp_model_coef[coef_col].values
        )
        tmp_model_coef = model_coef[model_coef[var_col].isin(impact_var)]
        for j in attribution_config["columns"]["random_effect_cols"]:
            temp = pd.DataFrame()
            temp = coef_Df[coef_Df[var_col].str.contains(j)]
            coef = temp.loc[temp[var_col].str.contains(i), coef_col].to_list()[0]
            tmp_model_coef.loc[tmp_model_coef.shape[0]] = [j, coef]
            model_coef.loc[model_coef.shape[0]] = [j, coef]
        impact_val = model_df[tmp_model_coef[var_col].to_list()].values.dot(
            tmp_model_coef[coef_col].values
        )

        model_df["baseline_contribution" + "_" + i] = np.exp(base_val)
        model_df["incremental_contribution" + "_" + i] = (
            model_df["Predicted_sales" + "_" + i]
            - model_df["baseline_contribution" + "_" + i]
        )

        # Calculate raw contribution for impact variables
        row_sum = 0
        abs_sum = 0
        for j in impact_var:
            tmp_impact_val = (
                model_df[j].values
                * model_coef.loc[model_coef[var_col] == j, coef_col].to_list()[0]
            )
            model_df[j + "_contribution_impact" + "_" + i] = np.exp(
                base_val + impact_val
            ) - np.exp(base_val + impact_val - tmp_impact_val)
            row_sum = row_sum + model_df[j + "_contribution_impact" + "_" + i]
            abs_sum = abs_sum + abs(model_df[j + "_contribution_impact" + "_" + i])

        y_b_s = model_df["incremental_contribution" + "_" + i] - row_sum
        impact_contribution = model_df[
            [attribution_config["time_column"], "Predicted_sales_" + i]
        ].copy()
        for j in impact_var:
            i_adj = j + "_contribution_impact" + "_" + i
            impact_contribution[i_adj] = (
                model_df[i_adj] + (abs(model_df[i_adj]) / abs_sum) * y_b_s
            )

        # Calculate raw contribution for base variables
        if attribution_config["model_type"] == "Mixed_effect":
            base_rc = (
                model_coef.loc[
                    model_coef[var_col] == intercept_name, coef_col
                ].to_list()[0]
                + rand_int_dict[i]
            )
        else:
            base_rc = model_coef.loc[
                model_coef[var_col] == intercept_name, coef_col
            ].to_list()[0]
        impact_contribution[intercept_name + "_contribution_base" + "_" + i] = np.exp(
            base_rc
        )
        for j in base_var[1:]:
            t = (
                tmp_model_df[j]
                * model_coef.loc[model_coef[var_col] == j, coef_col].to_list()[0]
                + base_rc
            )
            impact_contribution[j + "_contribution_base" + "_" + i] = np.exp(
                t
            ) - np.exp(base_rc)
            base_rc = t
        impact_contribution = impact_contribution.fillna(0)
        unit_dist_df = impact_contribution.copy()

        # Get Dollar Sales
        price_dist_df = unit_dist_df.copy()
        numeric_cols = price_dist_df.select_dtypes(include="number").columns.to_list()
        price_dist_df[numeric_cols] = price_dist_df[numeric_cols].mul(
            wk_sold_price.values, axis=0
        )

        # Get variable contribution variants
        dt_unit, qtr_unit, yr_unit = get_var_contribution_variants(
            unit_dist_df, "model_coefficient_name", "units", i, attribution_config
        )
        dt_unit_dist_df = pd.concat([dt_unit_dist_df, dt_unit], axis=0)
        qtr_unit_dist_df = pd.concat([qtr_unit_dist_df, qtr_unit], axis=0)
        yr_unit_dist_df = pd.concat([yr_unit_dist_df, yr_unit], axis=0)

        (dt_price, qtr_price, yr_price,) = get_var_contribution_variants(
            price_dist_df, "model_coefficient_name", "price", i, attribution_config
        )
        dt_price_dist_df = pd.concat([dt_price_dist_df, dt_price], axis=0)
        qtr_price_dist_df = pd.concat([qtr_price_dist_df, qtr_price], axis=0)
        yr_price_dist_df = pd.concat([yr_price_dist_df, yr_price], axis=0)

    overall_dt_dist_df = dt_unit_dist_df.merge(
        dt_price_dist_df,
        how="left",
        on=[attribution_config["time_column"], "model_coefficient_name"],
    )
    overall_dt_dist_df.drop(gv + "_y", axis="columns", inplace=True)
    overall_dt_dist_df.rename(columns={gv + "_x": gv}, inplace=True)
    overall_dt_dist_df["model_coefficient_name"] = (
        overall_dt_dist_df["model_coefficient_name"]
        .str.split("_")
        .str[:-1]
        .str.join("_")
    )

    overall_qtr_dist_df = qtr_unit_dist_df.merge(
        qtr_price_dist_df, how="left", on=["Quarter", "model_coefficient_name"]
    )
    overall_qtr_dist_df.drop(gv + "_y", axis="columns", inplace=True)
    overall_qtr_dist_df.rename(columns={gv + "_x": gv}, inplace=True)
    overall_qtr_dist_df["model_coefficient_name"] = (
        overall_qtr_dist_df["model_coefficient_name"]
        .str.split("_")
        .str[:-1]
        .str.join("_")
    )

    overall_yr_dist_df = yr_unit_dist_df.merge(
        yr_price_dist_df, how="left", on=["Year", "model_coefficient_name"]
    )
    overall_yr_dist_df.drop(gv + "_y", axis="columns", inplace=True)
    overall_yr_dist_df.rename(columns={gv + "_x": gv}, inplace=True)
    overall_yr_dist_df["model_coefficient_name"] = (
        overall_yr_dist_df["model_coefficient_name"]
        .str.split("_")
        .str[:-1]
        .str.join("_")
    )
    return overall_dt_dist_df, overall_qtr_dist_df, overall_yr_dist_df


def sales_qty_contr_sums(
    pred_contr_df: pd.DataFrame, attribution_config: dict, level: str
):
    """Return sales quantity contribution and sales dollars contribution.

    Parameters
    ----------
    pred_contr_df: pd.DataFrame
        Dataframe with predicted contribution
    attribution_config: dict
        Dictionary of attribution config from config file
    level: str
        Level at which sales contribution is calculated

    Returns
    -------
    pd.DataFrame
        Dataframe consisting of sales quantity contribution
    """
    pred_contr_df[attribution_config["time_column"]] = pd.to_datetime(
        pred_contr_df[attribution_config["time_column"]], format="%Y-%m-%d"
    )
    pred_contr_df["year"] = pred_contr_df[attribution_config["time_column"]].dt.year
    pred_contr_df["month"] = pred_contr_df[attribution_config["time_column"]].dt.month
    pred_contr_df["c_avg_price"] = pred_contr_df["sales"] / pred_contr_df["pos_qty"]

    variable_columns = attribution_config["columns"]["marketing_vars"]
    pred_contr_df.columns = pred_contr_df.columns.map(
        lambda x: x + "_qty_contr" if x in variable_columns else x
    )

    qty_cols = [x for x in pred_contr_df.columns.tolist() if "_qty_contr" in x]
    for c in qty_cols:
        sc = c.replace("qty_contr", "sales_contr")
        pred_contr_df[sc] = (
            pred_contr_df[c] * pred_contr_df["c_avg_price"]
        )  # Units * Avg price
    cols = pred_contr_df.columns.tolist()
    cols = [c for c in cols if ("qty_contr" in c) | ("sales_contr" in c)]

    overall_sums = pd.DataFrame()
    overall_sums["column"] = cols
    overall_sums["overall_contribution"] = overall_sums["column"].apply(
        lambda x: pred_contr_df[x].sum()
    )
    overall_sums = overall_sums.loc[overall_sums["column"].str.endswith("sales_contr")]
    return overall_sums


def spend_sums(attribution_config: dict, level: str):
    """Return aggregated spends of all marketing channels.

    Parameters
    ----------
    attribution_config: dict
        Dictionary of attribution config from config file
    level: str
        Level at which sales contribution is calculated

    Returns
    -------
    pd.DataFrame
        Dataframe consisting of total spend
    """
    spend_data = pd.read_csv(
        attribution_config["contributions"]["spends_file_location"]
    )
    spend_data[attribution_config["time_column"]] = pd.to_datetime(
        spend_data[attribution_config["time_column"]], infer_datetime_format=True
    )
    spend_data["year"] = spend_data[attribution_config["time_column"]].dt.year
    spend_data["month"] = spend_data[attribution_config["time_column"]].dt.month
    spend_data["quarter"] = spend_data[attribution_config["time_column"]].dt.quarter
    spend_data["quarter"] = "Q" + spend_data["quarter"].astype(str)

    if attribution_config["contributions"]["Level"] == "Monthly":
        spend_data["level"] = (
            spend_data["year"].astype(str) + "-" + spend_data["month"].astype(str)
        )
    elif attribution_config["contributions"]["Level"] == "Quarterly":
        spend_data["level"] = (
            spend_data["year"].astype(str) + "-" + spend_data["quarter"].astype(str)
        )
    elif attribution_config["contributions"]["Level"] == "Yearly":
        spend_data["level"] = spend_data["year"].astype(str)
    elif attribution_config["contributions"]["Level"] == "Overall":
        spend_data["level"] = "Overall"

    spend_data = spend_data.loc[spend_data["level"] == level, :]
    cols = spend_data.columns.tolist()
    cols = [c for c in cols if c != attribution_config["time_column"]]
    overall_sums = pd.DataFrame()
    overall_sums["column"] = cols
    overall_sums["overall_spend"] = overall_sums["column"].apply(
        lambda x: spend_data[x].sum()
    )
    return overall_sums


def get_spend(spend_data: pd.DataFrame, x: str, column_name: str):
    """Intermediate function to calculate spend.

    Parameters
    ----------
    spend_data: pd.DataFrame
        Spend dataset
    x: str
        Column in spend data
    column_name: str
        Variable for which spend is to be calculated

    Returns
    -------
    float
        Calculated spend for the variable
    """
    x = x.replace("_sales_contr", "")
    if x in spend_data["column"].tolist():
        s = spend_data.loc[spend_data["column"] == x, :][column_name]
        if s.shape[0] > 0:
            return s.tolist()[0]
    else:
        prefix = x.split(".")[0]
        spend_data["prefix"] = spend_data["column"].apply(lambda x: x.split(".")[0])
        s = spend_data.loc[spend_data["prefix"] == prefix, :][column_name]
        if s.shape[0] > 0:
            return s.tolist()[0]
        else:
            return None


def get_response_curves_data(
    quantity_contributions_my: pd.DataFrame, multipliers: list, transformation: str
):
    """Create response curves as dataframe.

    Parameters
    ----------
    quantity_contributions_my: pd.DataFrame
        Quantity contribution for max year
    multipliers: list
        List of multipliers
    transformation: str
        loglog or semilog transformation

    Returns
    -------
    pd.DataFrame
        Dataframe with response curve calculations
    """
    quantity_contributions_my["column"] = quantity_contributions_my[
        "column"
    ].str.replace("_csales_contr", "")
    cols = quantity_contributions_my["column"].unique().tolist()
    final_response_df = pd.DataFrame()
    for c in cols:
        response_df = pd.DataFrame()
        response_df["multiplier"] = multipliers
        response_df["touch_point"] = c
        response_df["beta"] = quantity_contributions_my.loc[
            quantity_contributions_my["column"] == c, "beta"
        ].tolist()[0]
        response_df["sales"] = quantity_contributions_my.loc[
            quantity_contributions_my["column"] == c, "overall_contribution"
        ].tolist()[0]
        response_df["spend"] = quantity_contributions_my.loc[
            quantity_contributions_my["column"] == c, "overall_spend"
        ].tolist()[0]
        response_df["new_spend"] = response_df["spend"] * (
            1 + response_df["multiplier"]
        )
        response_df["spend_change"] = response_df["new_spend"] - response_df["spend"]
        if transformation == "loglog":
            # response_df["new_sales"] = (((1 + response_df["new_spend"]) / response_df["spend"]) ** response_df["beta"]) * response_df["sales"]
            response_df["new_sales"] = (
                (response_df["new_spend"] / response_df["spend"]) ** response_df["beta"]
            ) * response_df["sales"]
        else:
            response_df["ratio"] = response_df["beta"] * (
                response_df["new_spend"] / response_df["spend"]
            )
            response_df["ratio"] = response_df["ratio"].apply(lambda x: math.exp(x) - 1)
            response_df["new_sales"] = response_df["ratio"] * response_df["sales"]
            response_df.drop("ratio", axis="columns", inplace=True)
        final_response_df = final_response_df.append(response_df)
    return final_response_df


def convert_to_req_format(contr_data: pd.DataFrame, attribution_config: dict):
    """Convert contribution data to required format.

    Parameters
    ----------
    contr_data: pd.DataFrame
        Calculated contribution data
    attribution_config: dict
        Dictionary of attribution config from config file

    Returns
    -------
    pd.DataFrame
        Contribution data in required format to calculate ROI
    """
    contr_data["model_coefficient_name"] = contr_data[
        "model_coefficient_name"
    ].str.replace("_contribution_impact", "")
    contr_data["model_coefficient_name"] = contr_data[
        "model_coefficient_name"
    ].str.replace("_contribution_base", "")
    data_roi = pd.DataFrame(columns=attribution_config["columns"]["marketing_vars"])
    n = data_roi.columns.nunique()
    length = [[]] * n
    for i, j in zip(data_roi.columns, range(0, n)):
        # print(i)
        # print(j)
        k = contr_data[contr_data["model_coefficient_name"] == i]["units"]
        length[j].append(k)
        se = pd.Series(k)
        data_roi[i] = se.values
    data_roi[attribution_config["time_column"]] = contr_data[
        attribution_config["time_column"]
    ]
    return data_roi


def calc_overall_roi(
    model_data: pd.DataFrame, data_roi: pd.DataFrame, attribution_config: dict
):
    """Calculate ROI.

    Parameters
    ----------
    model_data: pd.DataFrame
        Modelled data used to calculate contributions
    data_roi: pd.DataFrame
        Calculated contributions data
    attribution_config: dict
        Dictionary of attribution config from config file

    Returns
    -------
    pd.DataFrame
        Calculated overall ROI for each marketing variable
    """
    model_data.rename(
        columns={
            attribution_config["dv"]: "pos_qty",
            attribution_config["sales_dollars_column"]: "sales",
        },
        inplace=True,
    )
    # Merge model data with qty contribution to get sales_qty_contr
    pred_contr_df_sales_contr = pd.merge(
        data_roi,
        model_data[[attribution_config["time_column"], "pos_qty", "sales", "y_pred"]],
        on=attribution_config["time_column"],
        how="left",
    )

    m = attribution_config["contributions"]["Level"]
    sales_contributions_final = pd.DataFrame()
    sales_contributions = sales_qty_contr_sums(
        pred_contr_df_sales_contr, attribution_config, m
    )
    spend_data = spend_sums(attribution_config, m)
    sales_contributions["overall_spend"] = sales_contributions["column"].apply(
        lambda x: get_spend(spend_data, x, "overall_spend")
    )
    sales_contributions[f"ROI_{m}"] = (
        sales_contributions["overall_contribution"]
        / sales_contributions["overall_spend"]
    )

    if attribution_config["contributions"]["Level"] != "Overall":
        sales_contributions = sales_contributions.loc[:, ["column", f"ROI_{m}"]]
    if sales_contributions_final.shape[0] == 0:
        sales_contributions_final = sales_contributions
    else:
        sales_contributions_final = sales_contributions_final.merge(
            sales_contributions, on="column", how="outer"
        )
    return sales_contributions_final


def calc_response_curve_data(
    model_data: pd.DataFrame,
    qty_contr: pd.DataFrame,
    attribution_config: dict,
    coef_df: pd.DataFrame,
    transformation: str,
):
    """Calculate Response Curve.

    Parameters
    ----------
    model_data: pd.DataFrame
        Modelled data used to calculate contributions
    qty_contr: pd.DataFrame
        Calculated contributions data for max year
    attribution_config: dict
        Dictionary of attribution config from config file
    coef_df: pd.DataFrame
        Dataframe with beta coefficients of variables
    transformation: str
        loglog or semilog transformation

    Returns
    -------
    pd.DataFrame
        Dataframe with Response curve calculations
    """
    model_data.rename(
        columns={
            attribution_config["dv"]: "pos_qty",
            attribution_config["sales_dollars_column"]: "sales",
        },
        inplace=True,
    )
    model_data["year"] = model_data[attribution_config["time_column"]].dt.year
    model_data["year"] = model_data["year"].astype(int)
    max_year = model_data["year"].max()
    model_data_max_year = model_data.loc[model_data["year"] == max_year, :]

    qty_contr["year"] = qty_contr[attribution_config["time_column"]].dt.year
    qty_contr["year"] = qty_contr["year"].astype(int)
    max_year = qty_contr["year"].max()
    qty_contr_data_max_year = qty_contr.loc[qty_contr["year"] == max_year, :]

    # Merge model data with qty contribution to get sales_qty_contr
    pred_contr_df_sales_contr_max_year = pd.merge(
        qty_contr_data_max_year,
        model_data_max_year[
            [attribution_config["time_column"], "pos_qty", "sales", "y_pred"]
        ],
        on=attribution_config["time_column"],
        how="left",
    )

    sales_contributions_my = sales_qty_contr_sums(
        pred_contr_df_sales_contr_max_year,
        attribution_config,
        str(max_year),
    )
    spend_data_my = spend_sums(attribution_config, "2020")
    sales_contributions_my["overall_spend"] = sales_contributions_my["column"].apply(
        lambda x: get_spend(spend_data_my, x, "overall_spend")
    )
    sales_contributions_my["column"] = sales_contributions_my["column"].str.replace(
        "_sales_contr", ""
    )
    sales_contributions_my = sales_contributions_my.merge(
        coef_df, on="column", how="left"
    )

    # multipliers=[round(-0.9+(x*0.05),2) for x in range(0,37)]
    multipliers = [round(-0.9 + (x * 0.1), 2) for x in range(0, 30)]
    rcd = get_response_curves_data(sales_contributions_my, multipliers, transformation)
    return rcd


def get_attributions(
    data: pd.DataFrame,
    coef_df: pd.DataFrame,
    sales_dollar_column: pd.Series,
    var_col: str,
    coef_col: str,
    intercept_name: str,
    base_var: list,
    attribution_config: dict,
    global_config: dict,
    model_config: dict,
    out_path="",
):
    """Read config and get attributions of a model.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe consisting of model data
    coef_df: pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    sales_dollar_column: pd.Series
        Sales dollar column value
    var_col: str
        Column containing model variable
    coef_col: str
        Column containing model variables' coefficient estimate
    intercept_name: str
        Name of intercept variable
    base_var: list of str
        Base variables
    attribution_config: dict
        Attribution config from config file
    global_config: dict
        Global config from config file
    model_config: dict
        Model config from config file
    out_path: str
        Output path to save attribution results

    Returns
    -------
    Attributions for given dataset
    """
    d2 = datetime.now()
    d2 = d2.strftime("%m%d%Y_%H%M%S")
    try:
        d2 = datetime.now()
        d2 = d2.strftime("%m%d%Y_%H%M%S")

        # Applying log transformation on variables:
        # For log-log trasformation
        if model_config["type_of_model"]["transformation"] == "loglog":
            transformation = "loglog"
            for c in data.select_dtypes(include=[np.number]).columns:
                if c not in attribution_config["columns"]["categorical_vars"]:
                    data[c] = np.log(data[c])
        # For semi-log transformation
        else:
            transformation = "semilog"
            dv = global_config["dv"]
            data[dv] = np.log(data[dv])
        coef_df = coef_df.dropna()
        coef_df = coef_df[coef_df["column"] != "sigma_target"]

        # Calling above function
        (
            overall_dt_dist_df,
            overall_qtr_dist_df,
            overall_yr_dist_df,
        ) = get_var_contribution_wo_baseline_defined(
            data,
            coef_df,
            sales_dollar_column,
            var_col,
            coef_col,
            intercept_name,
            base_var,
            attribution_config,
            all_df=None,
        )

        # Writing output to file
        report_dict = {}
        wb = f'{out_path}/Attribution_results_{global_config["run_name"]}_{d2}.xlsx'
        writer = pd.ExcelWriter(wb)

        overall_dt_dist_df.to_excel(writer, "Datewise_Contribution", index=False)
        overall_qtr_dist_df.to_excel(writer, "Quarterly_Contribution", index=False)
        overall_yr_dist_df.to_excel(writer, "Yearly_Contribution", index=False)

        # Converting contribution data to required format before calculation ROI
        data_roi = convert_to_req_format(overall_dt_dist_df, attribution_config)

        # ROI Calculation
        attribution_config_overall = attribution_config.copy()
        attribution_config_overall["contributions"]["Level"] = "Overall"
        roi_data = calc_overall_roi(data, data_roi, attribution_config_overall)
        roi_data.rename(
            columns={
                "overall_spend": "spend",
                "ROI_Overall": "ROI",
                "overall_contribution": "sales",
            },
            inplace=True,
        )
        roi_data["spend_per"] = roi_data["spend"] / (roi_data["spend"].sum())
        roi_data["sales_per"] = roi_data["sales"] / (roi_data["sales"].sum())
        roi_data["Efficiency"] = roi_data["sales_per"] / roi_data["spend_per"]
        roi_data.to_excel(writer, "ROI", index=False)

        # Response Curve Calculation
        attribution_config_copy = attribution_config.copy()
        attribution_config_copy["contributions"]["Level"] = "Yearly"
        response_curve_data = calc_response_curve_data(
            data, data_roi, attribution_config_copy, coef_df, transformation
        )
        response_curve_data.to_excel(writer, "Response Curve Calculation", index=False)

        report_dict["Response Curves"] = {}
        cols = response_curve_data["touch_point"].unique().tolist()
        for c in cols:
            rcd_f = response_curve_data.loc[response_curve_data["touch_point"] == c, :]
            report_dict["Response Curves"][c] = {}
            report_dict["Response Curves"][c]["plot"] = hvPlot(rcd_f).line(
                x="spend_change",
                y=["new_sales"],
                label=f"Response Curve for {c}",
                width=700,
                height=500,
            )
        create_report(
            report_dict,
            name=f'Attributions_output_{global_config["run_name"]}_{d2}',
            path=out_path,
            format=".html",
            split_sheets=True,
            tiger_template=False,
        )
        writer.save()

    except Exception:
        logging.error("Exception occurred in Attribution calculations", exc_info=True)
