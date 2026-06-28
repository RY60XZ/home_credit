from lightgbm import LGBMClassifier
from src.preprocessing import make_model_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings

def make_logistic_regression():
    return make_model_pipeline(
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver="liblinear",
            random_state=42,
        ),
        scale_numeric = True,
    )

def make_random_forest():
    return make_model_pipeline(
        RandomForestClassifier(
            n_estimators = 300,
            min_samples_leaf = 20,
            max_depth = None,
            class_weight='balanced',
            random_state=42,
            n_jobs = -1,
        ),
        scale_numeric = False,
    )

def make_lightGBM(lgbm_params=None):
    params = {
        "n_estimators": 300,
        "min_child_samples": 20,
        "max_depth": None,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if lgbm_params is not None:
        params.update(lgbm_params)

    return make_model_pipeline(
        LGBMClassifier(
            **params
        ),
        scale_numeric = False,
    )

def evaluate_roc_auc(model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    return auc
