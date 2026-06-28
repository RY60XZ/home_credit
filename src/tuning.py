import json
import optuna
import pandas as pd
import warnings
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from optuna.trial import TrialState
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.preprocessing import make_preprocessor
from src.training import run_lgbm_cv
from src.data_loading import PROCESSED_DATA_PATH


def _sample_data(X, y, sample_size=None, random_state=42):
    if sample_size is None or sample_size >= len(X):
        return X, y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        random_state=random_state,
        stratify=y,
    )
    return X_sample, y_sample


def _trial_params(trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 96, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10]),
    }


def _params_from_trial(trial):
    params = trial.params.copy()
    params["subsample_freq"] = 1
    params["n_estimators"] = max(50, trial.user_attrs["best_iteration"])
    return params


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_lgbm_params(params, filename="best_lgbm_params.json"):
    path = PROCESSED_DATA_PATH / filename
    with path.open("w") as file:
        json.dump(_json_ready(params), file, indent=2)
    return path


def load_lgbm_params(filename="best_lgbm_params.json"):
    path = PROCESSED_DATA_PATH / filename
    with path.open() as file:
        return json.load(file)


def tune_lgbm_fast(
    X,
    y,
    n_trials=10,
    top_n=5,
    sample_size=75000,
    valid_size=0.2,
    early_stopping_rounds=30,
    random_state=42,
    quiet=True,
):
    if quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_sample, y_sample = _sample_data(
        X,
        y,
        sample_size=sample_size,
        random_state=random_state,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_sample,
        y_sample,
        test_size=valid_size,
        random_state=random_state,
        stratify=y_sample,
    )

    preprocessor = make_preprocessor(scale_numeric=False)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    fixed_params = {
        "objective": "binary",
        "n_estimators": 800,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }

    def objective(trial):
        params = fixed_params | _trial_params(trial)
        model = LGBMClassifier(**params)
        with warnings.catch_warnings():
            if quiet:
                warnings.simplefilter("ignore")
            model.fit(
                X_train_processed,
                y_train,
                eval_set=[(X_valid_processed, y_valid)],
                eval_metric="auc",
                callbacks=[
                    early_stopping(early_stopping_rounds, verbose=False),
                    log_evaluation(period=0),
                ],
            )
            valid_pred = model.predict_proba(X_valid_processed)[:, 1]
        auc = roc_auc_score(y_valid, valid_pred)
        trial.set_user_attr("best_iteration", model.best_iteration_)
        return auc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    trials = study.trials_dataframe()
    trials = trials.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]
    top_trials = sorted(complete_trials, key=lambda trial: trial.value, reverse=True)[:top_n]

    top_params = [_params_from_trial(trial) for trial in top_trials]
    if not top_params:
        raise ValueError("No completed tuning trials.")

    top_trials = pd.DataFrame(
        [
            {
                "candidate": i + 1,
                "trial_number": trial.number,
                "fast_auc": trial.value,
                "best_iteration": trial.user_attrs["best_iteration"],
            }
            for i, trial in enumerate(top_trials)
        ]
    )

    return {
        "best_auc": study.best_value,
        "best_params": top_params[0],
        "top_params": top_params,
        "top_trials": top_trials,
        "trials": trials,
        "preprocessor": preprocessor,
        "study": study,
    }


def confirm_lgbm_params_cv(X, y, lgbm_params, n_split=5, random_state=42):
    results = run_lgbm_cv(
        X,
        y,
        n_split=n_split,
        random_state=random_state,
        lgbm_params=lgbm_params,
    )
    return pd.DataFrame(
        {
            "mean_auc": [results["mean_auc"]],
            "std_auc": [results["std_auc"]],
            "oof_auc": [roc_auc_score(y, results["oof_preds"])],
        }
    ), results


def confirm_lgbm_candidates_cv(X, y, candidate_params, n_split=5, random_state=42):
    rows = []
    candidate_results = []

    for i, params in enumerate(candidate_params):
        auc_table, results = confirm_lgbm_params_cv(
            X,
            y,
            params,
            n_split=n_split,
            random_state=random_state,
        )
        row = auc_table.iloc[0].to_dict()
        row["candidate"] = i + 1
        rows.append(row)
        candidate_results.append(results)

    cv_table = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    return cv_table, candidate_results
