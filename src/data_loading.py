import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path.cwd()
if not (ROOT / "data" / "raw").exists():
    ROOT = ROOT.parent
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"


def _add_ratio_feature(application, new_col, numerator_col, denominator_col):
    if numerator_col in application.columns and denominator_col in application.columns:
        application[new_col] = application[numerator_col] / application[denominator_col]


def data_cleaning(application):
    application = application.copy()

    employed_anomaly = application["DAYS_EMPLOYED"] > 30000
    application.loc[employed_anomaly, "DAYS_EMPLOYED"] = np.nan

    application.loc[application["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan

    _add_ratio_feature(application, "CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL")
    _add_ratio_feature(application, "ANNUITY_INCOME_RATIO", "AMT_ANNUITY", "AMT_INCOME_TOTAL")
    _add_ratio_feature(application, "GOODS_CREDIT_RATIO", "AMT_GOODS_PRICE", "AMT_CREDIT")
    _add_ratio_feature(application, "CREDIT_ANNUITY_RATIO", "AMT_CREDIT", "AMT_ANNUITY")
    _add_ratio_feature(application, "EMPLOYMENT_AGE_RATIO", "DAYS_EMPLOYED", "DAYS_BIRTH")

    application = application.replace([np.inf, -np.inf], np.nan)
    return application


def load_data():
    application_train = pd.read_csv(RAW_DATA_PATH / "application_train.csv")
    application_test = pd.read_csv(RAW_DATA_PATH / "application_test.csv")
    application_train = data_cleaning(application_train)
    application_test = data_cleaning(application_test)
    return application_train, application_test


def train_valid_split(application_train):
    X = application_train.drop(columns=[TARGET_COL, ID_COL])
    y = application_train[TARGET_COL]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_valid, y_train, y_valid
