import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.feature_engineering import add_engineered_features

try:
    from src.data_loading import load_data, train_valid_split
except ModuleNotFoundError:
    from data_loading import load_data, train_valid_split


def get_feature_columns(X):
    categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    binary_columns = [col for col in numeric_columns if X[col].dropna().nunique() == 2]
    numeric_columns = [col for col in numeric_columns if col not in binary_columns]

    return categorical_columns, binary_columns, numeric_columns


def _categorical_columns(X):
    return get_feature_columns(X)[0]


def _binary_columns(X):
    return get_feature_columns(X)[1]


def _numeric_columns(X):
    return get_feature_columns(X)[2]


def make_numeric_pipeline(strategy="median", add_missing_indicators=True, scale=False):
    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy = strategy,
                add_indicator= add_missing_indicators,
            )
        )
    ]

    if scale:
        steps.append(("scaler", StandardScaler(with_mean=False)))

    return Pipeline(steps)


def make_binary_pipeline(strategy="most_frequent", add_missing_indicators=True):
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=strategy,
                    add_indicator=add_missing_indicators,
                ),
            )
        ]
    )


def make_categorical_pipeline(strategy="most_frequent"):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )


def make_preprocessor(scale_numeric=False, add_missing_indicators=True, categorical_encoding="onehot"):
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                make_numeric_pipeline(
                    add_missing_indicators=add_missing_indicators,
                    scale=scale_numeric,
                ),
                _numeric_columns,
            ),
            (
                "binary",
                make_binary_pipeline(
                    add_missing_indicators=add_missing_indicators,
                ),
                _binary_columns,
            ),
            (
                "categorical",
                make_categorical_pipeline(),
                _categorical_columns,
            ),
        ],
        remainder="drop",
    )


def make_model_pipeline(estimator, scale_numeric=False, add_missing_indicators=True, categorical_encoding="onehot"):
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    scale_numeric=scale_numeric,
                    add_missing_indicators=add_missing_indicators,
                    categorical_encoding=categorical_encoding,
                ),
            ),
            ("model", estimator),
        ]
    )


class LightGBMFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_prepared = add_engineered_features(X)
        self.feature_names_out_ = np.array(X_prepared.columns, dtype=object)
        self.categorical_columns_ = X_prepared.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        self.categories_ = {}
        for col in self.categorical_columns_:
            self.categories_[col] = pd.Series(X_prepared[col]).astype("category").cat.categories
        return self

    def transform(self, X):
        X_prepared = add_engineered_features(X)
        for col in self.feature_names_out_:
            if col not in X_prepared.columns:
                X_prepared[col] = np.nan
        X_prepared = X_prepared.loc[:, self.feature_names_out_].copy()

        for col in self.categorical_columns_:
            X_prepared[col] = pd.Categorical(
                X_prepared[col],
                categories=self.categories_[col],
            )
        return X_prepared

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def make_lightgbm_pipeline(estimator):
    return Pipeline(
        [
            ("preprocess", LightGBMFeaturePreprocessor()),
            ("model", estimator),
        ]
    )
