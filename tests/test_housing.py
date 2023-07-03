import pytest # noqa
from src.ta_lib.hyperparam_tuning import hyperparam as hp


def test_model():
    best_params = hp.tune_param('RandomForest')
    assert isinstance(best_params['r2_score'], float)
