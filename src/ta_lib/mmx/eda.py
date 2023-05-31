# isort: skip_file
import logging
import warnings
from datetime import datetime

import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models import GlyphRenderer, LinearAxis, Range1d
from bokeh.models.formatters import DatetimeTickFormatter
from hvplot import hvPlot

from tigerml.core.reports import create_report
from tigerml.eda import EDAReport

hv.extension("bokeh")

warnings.filterwarnings("ignore")


def max_date(x: str, time_column: str, data: pd.DataFrame):
    """Max date for a column.

    Parameters
    ----------
    x: str
        Column whose maximum date is to be calculated
    time_column: str
        Time_column from global_config
    data: pd.DataFrame
        Input Dataset

    Returns
    -------
    pd.Timestamp
        Maximum date of the column.
    """
    d = data.loc[:, [time_column, x]]
    d = d.dropna()
    return d[time_column].max()


def min_date(x: str, time_column: str, data: pd.DataFrame):
    """Min date for a column.

    Parameters
    ----------
    x: str
        Column whose minimum date to be calculated
    time_column: str
        Time_column from global_config
    data: pd.DataFrame
        Dataset

    Returns
    -------
    pd.Timestamp
        Minimum date of the column.
    """
    d = data.loc[:, [time_column, x]]
    d = d.dropna()
    return d[time_column].min()


def zero_counts(x: str, data: pd.DataFrame):
    """Count zeros in a column.

    Parameters
    ----------
    x: str
        Column whose zero count needs to be calculated
    data: pd.DataFrame
        Input Dataset

    Returns
    -------
    int
        Count of zeros in the column.
    """
    d = data.loc[data[x] == 0, :]
    return d.shape[0]


def univariate_analysis_numeric(data: pd.DataFrame, time_column: str):
    """Genreate univariate report for all the variables of a dataframe.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    time_column: str
        Time_column from global_config

    Returns
    -------
    dtypes : pd.DataFrame
        Dataframe with metrics of all numeric columns.
    """
    logging.info("Entering Univariate function for numeric cols")
    numeric_cols = data.select_dtypes([np.number]).columns.tolist()
    dtypes = data[numeric_cols].describe().T.reset_index()
    dtypes.rename(columns={"index": "column_name"}, inplace=True)
    dtypes["10%"] = dtypes["column_name"].apply(lambda x: data[x].quantile(0.1))
    dtypes["90%"] = dtypes["column_name"].apply(lambda x: data[x].quantile(0.9))
    dtypes["Missing_count"] = dtypes["column_name"].apply(
        lambda x: data[x].isna().sum()
    )
    dtypes["missing_per"] = dtypes["Missing_count"] / data.shape[0]
    dtypes.rename(
        columns={
            "count": "nRows_in_DataSet",
            "25%": "P25",
            "50%": "Median",
            "75%": "P75",
            "10%": "P10",
            "90%": "P90",
        },
        inplace=True,
    )
    dtypes["max_date"] = dtypes["column_name"].apply(
        lambda x: max_date(x, time_column, data)
    )
    dtypes["min_date"] = dtypes["column_name"].apply(
        lambda x: min_date(x, time_column, data)
    )
    dtypes["zero_counts"] = dtypes["column_name"].apply(lambda x: zero_counts(x, data))

    logging.info("coming out of univariate function for numeric cols")
    return dtypes


def univariate_analysis_categorical(data: pd.DataFrame, time_column: str):
    """Genreate univariate report for all the variables of a dataframe.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    time_column: str
        Time_column from global_config

    Returns
    -------
    ctypes : pd.DataFrame
        Dataframe with metrics of all categorical columns.
    """
    logging.info("Entering Univariate function for numeric cols")
    numeric_cols = data.select_dtypes([np.number]).columns.tolist()
    remaining_cols = [
        x for x in data.columns.tolist() if (x not in numeric_cols) & (x != time_column)
    ]
    ctypes = pd.DataFrame()
    if len(remaining_cols) > 0:
        ctypes["column_name"] = remaining_cols
        ctypes["Missing_count"] = ctypes["column_name"].apply(
            lambda x: data[x].isna().sum()
        )
        ctypes["missing_per"] = ctypes["Missing_count"] / data.shape[0]
        ctypes["n_categories"] = ctypes["column_name"].apply(
            lambda x: data[x].unique().shape[0]
        )
        ctypes["mode"] = ctypes["column_name"].apply(lambda x: data[x].mode().tolist())
        ctypes["max_date"] = ctypes["column_name"].apply(
            lambda x: max_date(x, time_column, data)
        )
        ctypes["min_date"] = ctypes["column_name"].apply(
            lambda x: min_date(x, time_column, data)
        )
        ctypes["zero_counts"] = ctypes["column_name"].apply(
            lambda x: zero_counts(x, data)
        )
    logging.info("coming out of univariate function for numeric cols")
    return ctypes


def return_outlier_ids(data: pd.DataFrame, time_column: str, col: str):
    """Identify outlier records.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    time_column: str
        Time_column from global_config
    col: str
        Numeric column for which outlier ids are calculated

    Returns
    -------
    list
        List of outlier ids of the column.
    """
    data = data.loc[:, [time_column, col]]
    mean = data[col].mean()
    std = data[col].std()
    data["z"] = (data[col] - mean) / std
    # upperlimit = mean + 3 * std
    # lowerlimit = mean - 3 * std
    data["o_i"] = np.where((data["z"] > 3) | (data["z"] < -3), "outlier", "no")
    return data.loc[data["o_i"] == "outlier", :]["date_sunday"].astype(str).tolist()


def identify_outliers(data: pd.DataFrame, time_column: str):
    """Identify outliers.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    time_column: str
        Time_column from global_config

    Returns
    -------
    pd.DataFrame
        Dataframe with all numeric columns and their outlier ids.
    """
    logging.info("entering identify outliers function")
    numeric_cols = data.select_dtypes([np.number]).columns.tolist()
    o_df = pd.DataFrame()
    o_df["column_name"] = numeric_cols
    o_df["outlier_ids"] = o_df["column_name"].apply(
        lambda x: return_outlier_ids(data, time_column, x)
    )
    logging.info("coming out of identify outliers function")
    return o_df


def correlation_analysis(data: pd.DataFrame, global_config: dict, out_path=""):
    """Correlation analysis.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset
    global_config: dict
        Global_config from config.yml
    out_path: str
        Output path to save correlation analysis results

    Returns
    -------
    Correlation analysis for all variables with dependent variable
    """
    logging.info("entering correlation analysis function")
    d1 = datetime.now()
    d1 = d1.strftime("%m%d%Y_%H%M%S")
    # wb = f'{out_path}/correlation_analysis_{global_config["run_name"]}_{d1}.xlsx'
    # writer = pd.ExcelWriter(wb)
    overall_corr = data.corr().reset_index()
    overall_corr = overall_corr.loc[:, ["index", global_config["dv"]]]
    # overall_corr.to_excel(writer, "Correlation with DV", index=False)
    # writer.close()
    logging.info("coming out of correlation analysis function")
    return overall_corr


def eda_report_generation(data: pd.DataFrame, global_config: dict, out_path=""):
    """EDA report using tigerml.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    global_config: dict
        Global_config from config.yml
    out_path: str
        Output path to save eda report

    Returns
    -------
    Generates EDA report using tigerml.
    """
    d1 = datetime.now()
    d1 = d1.strftime("%m%d%Y_%H%M%S")
    data[global_config["time_column"]] = data[global_config["time_column"]].astype(str)
    an = EDAReport(data, y=global_config["dv"])
    an.get_report(
        quick=True,
        name=f'eda_report_{global_config["run_name"]}_{d1}',
        save_path=out_path,
    )


def get_seasonality_column(data: pd.DataFrame, dv: str, level: str, time_column: str):
    """Add seasonality column.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    dv: str
        Dependent variable from global_config
    level: str
        Level in global_config file (week/month)
    time_column: str
        Time_column from global_config

    Returns
    -------
    pd.DataFrame
        Dataset with the seasonality column.
    """
    if level == "week":
        data[time_column] = pd.to_datetime(data[time_column])
        data["week"] = data[time_column].dt.week
        data_w = data.groupby("week", as_index=False).agg({dv: "mean"})
        avg = data[dv].mean()
        data_w[dv] = data_w[dv] / avg
        data_w.rename(columns={dv: "s_index_weekly"}, inplace=True)
        data = data.merge(data_w, on="week", how="left")
        data.drop("week", axis=1, inplace=True)
    elif level == "month":
        data[time_column] = pd.to_datetime(data[time_column])
        data["week"] = data[time_column].dt.month
        data_w = data.groupby("month", as_index=False).agg({dv: "mean"})
        avg = data[dv].mean()
        data_w[dv] = data_w[dv] / avg
        data_w.rename(columns={dv: "s_index_monthly"}, inplace=True)
        data = data.merge(data_w, on="month", how="left")
        data.drop("month", axis=1, inplace=True)
    return data


def add_trend_column(data: pd.DataFrame):
    """Add trend column.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset

    Returns
    -------
    pd.DataFrame
        Data by adding trend column.
    """
    data = data.reset_index(drop=True)
    data["trend"] = data.index + 1
    return data


def apply_formatter(plot, element):
    p = plot.state
    # create secondary range and axis
    p.extra_y_ranges = {"twiny": Range1d(start=200000, end=1200000)}
    p.add_layout(LinearAxis(y_range_name="twiny"), "left")
    # set glyph y_range_name to the one we've just created
    glyph = p.select(dict(type=GlyphRenderer))[0]
    glyph.y_range_name = "twiny"


def bivariate_plots(data: pd.DataFrame, global_config: dict, out_path=""):
    """Create bivariate report.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    global_config: dict
        Global_config from config.yml
    out_path: str
        Output path to save bivariate plots
    """
    d1 = datetime.now()
    d1 = d1.strftime("%m%d%Y_%H%M%S")
    numeric_cols = data.select_dtypes([np.number]).columns.tolist()
    dv = global_config["dv"]
    tc = global_config["time_column"]
    # data[tc] = data[tc].astype(str)
    data[tc] = data[tc].astype("datetime64[ns]")
    numeric_cols = [x for x in numeric_cols if x not in [tc, dv]]
    plot_dict = {"Bivariate Plots": {}}
    for i in numeric_cols:
        # fig = make_subplots(specs=[[{"secondary_y": True}]])
        plot = hvPlot(data).line(
            x=tc,
            y=[i, dv],
            legend="top",
            height=500,
            width=950,
            by=["index.year", "index.month"],
        )
        data_f = data.loc[:, [tc, i, dv]]
        data_f = data_f.set_index(tc)
        # fig.add_trace(go.Scatter(x=data_f[tc], y=data_f[i], name=i), secondary_y=False)
        # fig.add_trace(go.Scatter(x=data_f[tc], y=data_f[dv], name=dv), secondary_y=True)
        # fig.update_layout(
        #    height=500,
        #    width=950,
        #    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        # )
        a = data_f[i].hvPlot(yaxis="right", ylim=(data_f[i].min(), data_f[i].max()))
        b = (
            data_f[dv]
            .hvPlot(yaxis="left", ylim=(data_f[dv].min(), data_f[dv].max()))
            .opts(hooks=[apply_formatter])
        )
        plot = a * b
        plot = plot.opts(
            legend_position="top_left",
            xrotation=90,
            width=950,
            height=500,
            framewise=True,
            xticks=10,
            yticks=10,
            xformatter=DatetimeTickFormatter(months="%b %Y"),
        )
        plot_dict["Bivariate Plots"][i] = {}
        plot_dict["Bivariate Plots"][i]["plot"] = plot
    create_report(
        plot_dict,
        name=f'bivariate_plots_{global_config["run_name"]}_{d1}',
        path=out_path,
        format=".html",
        split_sheets=True,
        tiger_template=False,
    )


def univariate_analysis_main(
    data: pd.DataFrame, global_config: dict, d1: str, out_path=""
):
    """
    Univariate analysis.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    global_config: dict
        Global config from config.yml
    d1: str
        Current date
    out_path: str
        Output path to save univariate analysis
    """
    wb = f'{out_path}/univariate_analysis_{global_config["run_name"]}_{d1}.xlsx'
    writer = pd.ExcelWriter(wb)
    x = univariate_analysis_numeric(data, global_config["time_column"])
    x.to_excel(writer, "Numeric_cols_overall", index=False)
    x = pd.DataFrame()
    for i in data[global_config["gv"]].unique():
        # data_i = data[data[global_config["gv"]] == i]
        u = univariate_analysis_numeric(data, global_config["time_column"])
        u[global_config["gv"]] = i
        if x.empty:
            x = u
        else:
            x = pd.concat([x, u], axis=0)
    logging.info("saving univariate results of numeric cols to file")
    x.to_excel(writer, "Numeric_cols_categorywise", index=False)
    y = univariate_analysis_categorical(data, global_config["time_column"])
    if y.shape[0] > 0:
        logging.info("saving univariate results of categorical cols to file")
        y.to_excel(writer, "Categorical_cols", index=False)
    writer.close()
    return data


def correlation_analysis_main(
    data: pd.DataFrame, global_config: dict, d1: str, out_path=""
):
    """
    Correlation analysis.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    global_config: dict
        Global config from config.yml
    d1: str
        Current date
    out_path: str
        Output path to save correlation analysis
    """
    wb = f'{out_path}/correlation_analysis_{global_config["run_name"]}_{d1}.xlsx'
    writer = pd.ExcelWriter(wb)
    overall_corr = correlation_analysis(data, global_config, out_path)
    overall_corr.to_excel(writer, "Overall_Corr_with_DV", index=False)
    overall_corr = pd.DataFrame()
    for i in data[global_config["gv"]].unique():
        # data_i = data[data[global_config["gv"]] == i]
        oc = correlation_analysis(data, global_config, out_path)
        oc[global_config["gv"]] = i
        if overall_corr.empty:
            overall_corr = oc
        else:
            overall_corr = pd.concat([overall_corr, oc], axis=0)
    logging.info("saving correlation results to file")
    overall_corr.to_excel(writer, "Categorywise_Corr_with_DV", index=False)
    writer.close()
    return data


def outlier_analysis_main(
    data: pd.DataFrame, global_config: dict, d1: str, out_path=""
):
    """Outlier analysis.

    Parameters
    ----------
    data: pd.DataFrame
        Input Dataset
    global_config: dict
        Global config from config.yml
    d1: str
        Current date
    out_path: str
        Output path to save outlier analysis
    """
    wb = f'{out_path}/outlieres_{global_config["run_name"]}_{d1}.xlsx'
    writer = pd.ExcelWriter(wb)
    outliers = identify_outliers(data, global_config["time_column"])
    outliers.to_excel(writer, "Outliers_Overall", index=False)
    outliers = pd.DataFrame()
    for i in data[global_config["gv"]].unique():
        data_i = data[data[global_config["gv"]] == i]
        o = identify_outliers(data_i, global_config["time_column"])
        o[global_config["gv"]] = i
        if outliers.empty:
            outliers = o
        else:
            outliers = pd.concat([outliers, o], axis=0)
    logging.info("saving outlier detection results to file")
    outliers.to_excel(writer, "Outliers_categorywise", index=False)
    writer.close()
    return data


def data_analysis(data: pd.DataFrame, global_config: dict, out_path=""):
    """EDA execution wrapper.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataset
    global_config: dict
        Global_config from config.yml
    out_path: str
        Output path to store eda output

    Returns
    -------
    pd.DataFrame
        Clean data ready for feature transformation.
    """
    try:
        if global_config["seasonality"]["add_col"]:
            logging.info("Adding seasonality column")
            data = get_seasonality_column(
                data,
                global_config["dv"],
                global_config["seasonality"]["level"],
                global_config["time_column"],
            )
    except Exception:
        logging.error(
            "Exception occurred while adding seasonality column", exc_info=True
        )

    try:
        if global_config["add_trend_column"]:
            logging.info("Adding trend column")
            data = add_trend_column(data)
    except Exception:
        logging.error("Exception occurred while adding trend column", exc_info=True)

    d1 = datetime.now()
    d1 = d1.strftime("%m%d%Y_%H%M%S")

    try:
        step = 0
        # Getting univariate analysis
        data = univariate_analysis_main(data, global_config, d1, out_path)
        step = 1
        # EDA report generation
        eda_report_generation(data, global_config, out_path)
        step = 2
        # Correlation analysis
        data = correlation_analysis_main(data, global_config, d1, out_path)
        step = 3
        # Outlier Analysis
        data = outlier_analysis_main(data, global_config, d1, out_path)
        step = 4
        # Bivariate Plots
        bivariate_plots(data, global_config, out_path)
        step = 5
    except Exception:
        dict_err_stage = {
            0: "running Univariate analysis",
            1: "saving tigerml eda report",
            2: "running correlation analysis",
            3: "identifying outliers",
            4: "bivaraite_plots function",
            5: "all good",
        }
        logging.error(f"Exception occurred while {dict_err_stage[step]}", exc_info=True)
    return data
