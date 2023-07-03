import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ta_lib.housing import custom_transform as ct


def data_prep(data):
    """data_prep.

    function returns the dataframe with creating all required features after
    transforming into a single dataframe.

    Parameters
    ----------
    data : Pandas Dataframe
           Housing train dataset
    Returns
    ----------
    housing_prepared : Pandas Dataframe
                       Customized features set
    """
    housing = data.copy()
    indices = []
    cust_cols = ['total_rooms', 'total_bedrooms', 'population', 'households']
    for i in cust_cols:
        indices.append((housing.columns.get_loc(i)))
    housing.drop(["income_cat"], axis=1, inplace=True)
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    num_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', ct.CombinedAttributesAdder(indices, True)),
            ('std_scaler', StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
        ]
    )
    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = [housing.columns.get_loc(c) for c in col_names]
    housing_prepared = full_pipeline.fit_transform(housing)

    extra_attribs = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    attr_adder_columns = list(
        full_pipeline.named_transformers_["cat"].get_feature_names_out(
            ["ocean_proximity"]
        )
    )
    full_pipeline_columns = num_attribs + extra_attribs + attr_adder_columns
    housing_prepared = pd.DataFrame(housing_prepared, columns=full_pipeline_columns, index=housing.index)
    return housing_prepared
