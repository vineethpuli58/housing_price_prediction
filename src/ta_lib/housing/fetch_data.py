import os
import tarfile
import urllib
import pandas as pd


def fetch_housing_data(housing_url, housing_path):
    """fetch_housing_data.

    function downloads the data and saves in respective folder.

    Parameters
    ----------
    housing_path: str
                  Path to save the datasets
    housing_url:  str
                  URL to download the data
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_data(housing_path):
    """load_data.

    load_data function loads the data to the data/raw folder in
    the main folder structure.

    Parameters
    ----------
    housing_path: str
                  Path to save the datasets
    """
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    # HERE = op.dirname(op.abspath(__file__))
    HOUSING_PATH = housing_path
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)


def load_housing_data(housing_path):
    """load_housing_data.

    This function loads the data from the in put path and
    returns the pandas dataframe.

    Parameters
    ----------
    housing_path : str
                   Path to save the datasets
    Returns
    -------
    df : pd.DataFrame
         data_frame from the input file.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    df = pd.read_csv(csv_path)
    return df
