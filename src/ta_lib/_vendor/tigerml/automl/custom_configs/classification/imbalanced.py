import numpy as np

config = {
    # Classifiers
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "class_weight": ["balanced"],
    },
    "sklearn.ensemble.ExtraTreesClassifier": {
        "n_estimators": [100],
        "criterion": ["gini", "entropy"],
        "max_features": np.arange(0.05, 1.01, 0.05),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "bootstrap": [True, False],
        "class_weight": ["balanced"],
    },
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": [100],
        "criterion": ["gini", "entropy"],
        "max_features": np.arange(0.05, 1.01, 0.05),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "bootstrap": [True, False],
        "class_weight": ["balanced"],
    },
    "sklearn.svm.LinearSVC": {
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "dual": [True, False],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "class_weight": ["balanced"],
    },
    "sklearn.linear_model.LogisticRegression": {
        "penalty": ["l1", "l2"],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "dual": [True, False],
        "class_weight": ["balanced"],
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": [100],
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "subsample": np.arange(0.05, 1.01, 0.05),
        "max_features": np.arange(0.05, 1.01, 0.05),
    },
    "xgboost.XGBClassifier": {
        "n_estimators": [100],
        "max_depth": range(1, 11),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "subsample": np.arange(0.05, 1.01, 0.05),
        "min_child_weight": range(1, 21),
        "nthread": [1],
    },
    # Preprocesssors
    "sklearn.preprocessing.Binarizer": {"threshold": np.arange(0.0, 1.01, 0.05)},
    "sklearn.preprocessing.MinMaxScaler": {},
    "sklearn.preprocessing.Normalizer": {"norm": ["l1", "l2", "max"]},
    "sklearn.decomposition.PCA": {
        "svd_solver": ["randomized"],
        "iterated_power": range(1, 11),
    },
    "tpot.builtins.ZeroCount": {},
    # Selectors
    "sklearn.feature_selection.SelectFwe": {
        "alpha": np.arange(0, 0.05, 0.001),
        "score_func": {"sklearn.feature_selection.f_classif": None},
    },
    "sklearn.feature_selection.SelectPercentile": {
        "percentile": range(1, 100),
        "score_func": {"sklearn.feature_selection.f_classif": None},
    },
    "sklearn.feature_selection.VarianceThreshold": {
        "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    "sklearn.feature_selection.RFE": {
        "step": np.arange(0.05, 1.01, 0.05),
        "estimator": {
            "sklearn.ensemble.ExtraTreesClassifier": {
                "n_estimators": [100],
                "criterion": ["gini", "entropy"],
                "max_features": np.arange(0.05, 1.01, 0.05),
            }
        },
    },
    "sklearn.feature_selection.SelectFromModel": {
        "threshold": np.arange(0, 1.01, 0.05),
        "estimator": {
            "sklearn.ensemble.ExtraTreesClassifier": {
                "n_estimators": [100],
                "criterion": ["gini", "entropy"],
                "max_features": np.arange(0.05, 1.01, 0.05),
            }
        },
    },
}
