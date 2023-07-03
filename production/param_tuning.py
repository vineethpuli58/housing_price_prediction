from ta_lib.hyperparam_tuning import hyperparam
from ta_lib.core.api import (
    register_processor
)


@register_processor("param_tuning", "hyper_tuning")
def hyper_tuning(context, params):
    best_params = hyperparam.tune_param(params["hyperparam"])
    print(best_params,'best_params')
