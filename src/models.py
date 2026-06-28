from lightgbm import LGBMClassifier
from src.preprocessing import make_model_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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

def make_lightGBM():
    return make_model_pipeline(
        LGBMClassifier(
            n_estimators = 300,
            min_child_samples = 20,
            max_depth = None,
            class_weight='balanced',
            random_state=42,
            n_jobs = -1,
            verbose = -1,
        ),
        scale_numeric = False,
    )

def evaluate_roc_auc(model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    return auc