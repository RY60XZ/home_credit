import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loading import load_data, TARGET_COL, ID_COL
from src.training import run_lgbm_cv
from sklearn.metrics import roc_auc_score

def get_feature_importance(models, importance_type="gain"):
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        fold_importance = pd.DataFrame(
            {
                "feature": model.named_steps["preprocess"].get_feature_names_out(),
                f"fold_{i+1}": model.named_steps["model"].booster_.feature_importance(importance_type=importance_type),
            }
        )
        if feature_importance.empty:
            feature_importance = fold_importance
        else:
            feature_importance = feature_importance.merge(fold_importance, on="feature", how="outer")
    fold_cols = [col for col in feature_importance.columns if col.startswith("fold_")]
    feature_importance[fold_cols] = feature_importance[fold_cols].fillna(0)
    feature_importance["mean_importance"] = feature_importance[fold_cols].mean(axis=1)
    feature_importance["std_importance"] = feature_importance[fold_cols].std(axis=1)
    return feature_importance.sort_values("mean_importance", ascending=False).reset_index()

def plot_feature_importance(feature_importance, top_n=20):
    top_features = feature_importance.head(top_n).sort_values("mean_importance")
    fig, axis = plt.subplots(1, 1, figsize=(10, 5))
    axis.barh(top_features["feature"], top_features["mean_importance"])
    axis.set_title(f"Top {top_n} Feature Importances")
    axis.set_xlabel("Mean Importance")
    axis.set_ylabel("Feature")
    fig.tight_layout()
    return fig, axis

def summarize_cv_results(results, y):
    print(f"mean_auc: {results["mean_auc"]}")
    print(f"std_auc: {results["std_auc"]}")
    oof_auc = roc_auc_score(y, results["oof_preds"])
    print(f"oof_auc: {oof_auc}")

def get_error_cases(X, y, oof_preds, threshold=0.5):
    errors = X.copy()
    errors["y_true"] = y.values
    errors["oof_pred"] = oof_preds
    errors["y_pred"] = (errors["oof_pred"] >= threshold).astype(int)
    false_positive = errors[
        (errors["y_true"] == 0) & (errors["y_pred"] == 1)
        ].sort_values("oof_pred", ascending=False)
    false_negative = errors[
        (errors["y_true"] == 1) & (errors["y_pred"] == 0)
        ].sort_values("oof_pred", ascending=True)
    return false_positive, false_negative
