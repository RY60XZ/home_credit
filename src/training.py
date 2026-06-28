from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.models import make_lightGBM
from src.data_loading import load_data, TARGET_COL, ID_COL
import numpy as np
import pandas as pd
import warnings

def run_lgbm_cv(X, y, n_split=5, random_state=42, lgbm_params=None):
    cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))
    fold_aucs = []
    models = []
    for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        lightgbm_model = make_lightGBM(lgbm_params=lgbm_params)
        lightgbm_model.fit(X_train, y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid_pred = lightgbm_model.predict_proba(X_valid)[:, 1]
        oof_preds[valid_index] = valid_pred
        auc_score = roc_auc_score(y_valid, valid_pred)
        fold_aucs.append(auc_score)
        models.append(lightgbm_model)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    return {
        "fold_aucs": fold_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "oof_preds": oof_preds,
        "models": models,
    }
