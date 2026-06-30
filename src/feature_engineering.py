import numpy as np


def _safe_divide(numerator, denominator):
    return numerator / denominator.mask(denominator == 0)


def _add_ratio_feature(data, new_col, numerator_col, denominator_col):
    if numerator_col in data.columns and denominator_col in data.columns:
        data[new_col] = _safe_divide(data[numerator_col], data[denominator_col])


def add_engineered_features(data):
    data = data.copy()

    ext_source_cols = [
        col
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if col in data.columns
    ]
    if ext_source_cols:
        data["EXT_SOURCE_MEAN"] = data[ext_source_cols].mean(axis=1)
        data["EXT_SOURCE_STD"] = data[ext_source_cols].std(axis=1)
        data["EXT_SOURCE_MIN"] = data[ext_source_cols].min(axis=1)
        data["EXT_SOURCE_MAX"] = data[ext_source_cols].max(axis=1)
        data["EXT_SOURCE_RANGE"] = data["EXT_SOURCE_MAX"] - data["EXT_SOURCE_MIN"]
        data["EXT_SOURCE_COUNT"] = data[ext_source_cols].notna().sum(axis=1)
        data["EXT_SOURCE_MISSING_COUNT"] = data[ext_source_cols].isna().sum(axis=1)

    if set(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]).issubset(data.columns):
        data["EXT_SOURCE_PROD"] = (
            data["EXT_SOURCE_1"] * data["EXT_SOURCE_2"] * data["EXT_SOURCE_3"]
        )
        for left, right in [("1", "2"), ("1", "3"), ("2", "3")]:
            left_col = f"EXT_SOURCE_{left}"
            right_col = f"EXT_SOURCE_{right}"
            data[f"EXT_SOURCE_{left}_{right}_PROD"] = data[left_col] * data[right_col]
            data[f"EXT_SOURCE_{left}_{right}_DIFF"] = data[left_col] - data[right_col]
            data[f"EXT_SOURCE_{left}_{right}_RATIO"] = _safe_divide(
                data[left_col],
                data[right_col],
            )

    for col in [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
    ]:
        if col in data.columns:
            data[f"{col.removeprefix('DAYS_')}_YEARS"] = -data[col] / 365.25

    ratio_features = [
        ("PAYMENT_RATE", "AMT_ANNUITY", "AMT_CREDIT"),
        ("INCOME_CREDIT_RATIO", "AMT_INCOME_TOTAL", "AMT_CREDIT"),
        ("INCOME_PER_PERSON", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        ("CREDIT_PER_PERSON", "AMT_CREDIT", "CNT_FAM_MEMBERS"),
        ("ANNUITY_PER_PERSON", "AMT_ANNUITY", "CNT_FAM_MEMBERS"),
        ("GOODS_PRICE_PER_PERSON", "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS"),
        ("CHILDREN_RATIO", "CNT_CHILDREN", "CNT_FAM_MEMBERS"),
        ("DEF_30_SOCIAL_RATIO", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"),
        ("DEF_60_SOCIAL_RATIO", "DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"),
        ("BUREAU_DEBT_CREDIT_SUM_RATIO", "BUREAU_AMT_CREDIT_SUM_DEBT_SUM", "BUREAU_AMT_CREDIT_SUM_SUM"),
        ("BUREAU_DEBT_CREDIT_MEAN_RATIO", "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN", "BUREAU_AMT_CREDIT_SUM_MEAN"),
        ("BUREAU_OVERDUE_CREDIT_SUM_RATIO", "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM", "BUREAU_AMT_CREDIT_SUM_SUM"),
        ("BUREAU_ANNUITY_CREDIT_SUM_RATIO", "BUREAU_AMT_ANNUITY_SUM", "BUREAU_AMT_CREDIT_SUM_SUM"),
        ("PREVIOUS_APPLICATION_CREDIT_SUM_RATIO", "PREVIOUS_AMT_APPLICATION_SUM", "PREVIOUS_AMT_CREDIT_SUM"),
        ("PREVIOUS_GOODS_CREDIT_SUM_RATIO", "PREVIOUS_AMT_GOODS_PRICE_SUM", "PREVIOUS_AMT_CREDIT_SUM"),
        ("PREVIOUS_DOWN_PAYMENT_CREDIT_SUM_RATIO", "PREVIOUS_AMT_DOWN_PAYMENT_SUM", "PREVIOUS_AMT_CREDIT_SUM"),
        ("INSTALLMENTS_PAYMENT_INSTALMENT_SUM_RATIO", "INSTALLMENTS_AMT_PAYMENT_SUM", "INSTALLMENTS_AMT_INSTALMENT_SUM"),
        ("INSTALLMENTS_PAYMENT_DIFF_INSTALMENT_SUM_RATIO", "INSTALLMENTS_PAYMENT_DIFFERENCE_SUM", "INSTALLMENTS_AMT_INSTALMENT_SUM"),
        ("CREDIT_CARD_BALANCE_LIMIT_SUM_RATIO", "CREDIT_CARD_AMT_BALANCE_SUM", "CREDIT_CARD_AMT_CREDIT_LIMIT_ACTUAL_SUM"),
        ("CREDIT_CARD_DRAWINGS_LIMIT_SUM_RATIO", "CREDIT_CARD_AMT_DRAWINGS_CURRENT_SUM", "CREDIT_CARD_AMT_CREDIT_LIMIT_ACTUAL_SUM"),
        ("CREDIT_CARD_PAYMENT_MIN_SUM_RATIO", "CREDIT_CARD_AMT_PAYMENT_CURRENT_SUM", "CREDIT_CARD_AMT_INST_MIN_REGULARITY_SUM"),
        ("POS_CASH_FUTURE_INSTALMENT_MEAN_RATIO", "POS_CASH_CNT_INSTALMENT_FUTURE_MEAN", "POS_CASH_CNT_INSTALMENT_MEAN"),
        ("POS_CASH_FUTURE_INSTALMENT_SUM_RATIO", "POS_CASH_CNT_INSTALMENT_FUTURE_SUM", "POS_CASH_CNT_INSTALMENT_SUM"),
    ]
    for new_col, numerator_col, denominator_col in ratio_features:
        _add_ratio_feature(data, new_col, numerator_col, denominator_col)

    if "AMT_CREDIT" in data.columns and "AMT_GOODS_PRICE" in data.columns:
        data["CREDIT_GOODS_DIFF"] = data["AMT_CREDIT"] - data["AMT_GOODS_PRICE"]

    if "AMT_INCOME_TOTAL" in data.columns and "AMT_ANNUITY" in data.columns:
        data["INCOME_ANNUITY_DIFF"] = data["AMT_INCOME_TOTAL"] - data["AMT_ANNUITY"]

    if "REGION_RATING_CLIENT" in data.columns and "REGION_RATING_CLIENT_W_CITY" in data.columns:
        data["REGION_RATING_MEAN"] = data[
            ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]
        ].mean(axis=1)
        data["REGION_RATING_DIFF"] = (
            data["REGION_RATING_CLIENT"] - data["REGION_RATING_CLIENT_W_CITY"]
        )

    document_cols = [col for col in data.columns if col.startswith("FLAG_DOCUMENT_")]
    if document_cols:
        data["DOCUMENT_FLAG_COUNT"] = data[document_cols].sum(axis=1)

    contact_cols = [
        col
        for col in [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        ]
        if col in data.columns
    ]
    if contact_cols:
        data["CONTACT_FLAG_COUNT"] = data[contact_cols].sum(axis=1)

    if "DAYS_BIRTH" in data.columns:
        for ext_source_col in ext_source_cols:
            data[f"{ext_source_col}_BIRTH_PROD"] = (
                data[ext_source_col] * data["DAYS_BIRTH"]
            )
            if "AMT_CREDIT" in data.columns:
                data[f"{ext_source_col}_CREDIT_PROD"] = (
                    data[ext_source_col] * data["AMT_CREDIT"]
                )
            if "AMT_ANNUITY" in data.columns:
                data[f"{ext_source_col}_ANNUITY_PROD"] = (
                    data[ext_source_col] * data["AMT_ANNUITY"]
                )

    return data.replace([np.inf, -np.inf], np.nan)
