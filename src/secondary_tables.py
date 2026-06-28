from src.data_loading import RAW_DATA_PATH, ID_COL, load_data
import pandas as pd

def _safe_divide(numerator, denominator):
    return numerator / denominator.mask(denominator == 0)


def _clean_status_value(value):
    return str(value).upper().replace(" ", "_").replace("-", "_").replace("/", "_")


def _make_status_features(data, group_col, status_col, prefix):
    status_counts = pd.crosstab(data[group_col], data[status_col])
    status_counts.columns = [
        f"{prefix}_{_clean_status_value(status)}_COUNT"
        for status in status_counts.columns
    ]

    count_cols = status_counts.columns.tolist()
    status_total = status_counts[count_cols].sum(axis=1).mask(lambda s: s == 0)
    for col in count_cols:
        status_counts[col.replace("_COUNT", "_RATIO")] = status_counts[col] / status_total

    return status_counts.reset_index()


def add_bureau(application_train, application_test, bureau):

    numeric_cols = [
        col
        for col in bureau.select_dtypes(include="number").columns
        if col not in [ID_COL, "SK_ID_BUREAU"]
    ]
    categorical_cols = bureau.select_dtypes(exclude="number").columns.tolist()

    agg_dict = {
        col: ["mean", "min", "max", "sum", "std"]
        for col in numeric_cols
    }
    agg_dict.update(
        {
            col: ["nunique"]
            for col in categorical_cols
        }
    )

    aggregated_bureau = bureau.groupby(ID_COL).agg(agg_dict)
    aggregated_bureau.columns = [
        f"BUREAU_{col}_{agg}".upper()
        for col, agg in aggregated_bureau.columns
    ]
    aggregated_bureau = aggregated_bureau.reset_index()

    bureau_status_features = _make_status_features(
        bureau,
        ID_COL,
        "CREDIT_ACTIVE",
        "BUREAU_CREDIT_ACTIVE",
    )
    aggregated_bureau = aggregated_bureau.merge(
        bureau_status_features,
        on=ID_COL,
        how="left",
    )

    bureau_counts = (
        bureau.groupby(ID_COL)
        .size()
        .reset_index(name="BUREAU_RECORD_COUNT")
    )
    aggregated_bureau = aggregated_bureau.merge(
        bureau_counts,
        on=ID_COL,
        how="left",
    )

    application_train = application_train.merge(
        aggregated_bureau,
        on=ID_COL,
        how="left",
    )
    application_test = application_test.merge(
        aggregated_bureau,
        on=ID_COL,
        how="left",
    )

    return application_train, application_test

def add_bureau_balance(bureau):
    bureau_balance = pd.read_csv(RAW_DATA_PATH / "bureau_balance.csv")
    bureau = bureau.copy()
    bureau_id = "SK_ID_BUREAU"
    status_col = "STATUS"

    status_counts = pd.crosstab(
        bureau_balance[bureau_id],
        bureau_balance[status_col],
    ).reset_index()

    statuses = ["0", "1", "2", "3", "4", "5", "C", "X"]
    status_counts = status_counts.rename(
        columns={status: f"BUREAU_BALANCE_STATUS_{status}_COUNT" for status in statuses}
    )
    for status in statuses:
        count_col = f"BUREAU_BALANCE_STATUS_{status}_COUNT"
        if count_col not in status_counts.columns:
            status_counts[count_col] = 0

    count_cols = [f"BUREAU_BALANCE_STATUS_{status}_COUNT" for status in statuses]
    month_count = status_counts[count_cols].sum(axis=1)
    safe_month_count = month_count.replace(0, pd.NA)

    status_counts["BUREAU_BALANCE_MONTH_COUNT"] = month_count
    status_counts["BUREAU_BALANCE_STATUS_0_RATIO"] = status_counts["BUREAU_BALANCE_STATUS_0_COUNT"] / safe_month_count
    status_counts["BUREAU_BALANCE_STATUS_C_RATIO"] = status_counts["BUREAU_BALANCE_STATUS_C_COUNT"] / safe_month_count
    status_counts["BUREAU_BALANCE_STATUS_X_RATIO"] = status_counts["BUREAU_BALANCE_STATUS_X_COUNT"] / safe_month_count
    status_counts["BUREAU_BALANCE_DPD_RATIO"] = (
        status_counts[[f"BUREAU_BALANCE_STATUS_{status}_COUNT" for status in ["1", "2", "3", "4", "5"]]].sum(axis=1)
        / safe_month_count
    )
    status_counts["BUREAU_BALANCE_SEVERE_DPD_RATIO"] = (
        status_counts[[f"BUREAU_BALANCE_STATUS_{status}_COUNT" for status in ["3", "4", "5"]]].sum(axis=1)
        / safe_month_count
    )

    bureau = bureau.merge(
        status_counts[
            [
                bureau_id,
                "BUREAU_BALANCE_MONTH_COUNT",
                "BUREAU_BALANCE_STATUS_0_RATIO",
                "BUREAU_BALANCE_STATUS_C_RATIO",
                "BUREAU_BALANCE_STATUS_X_RATIO",
                "BUREAU_BALANCE_DPD_RATIO",
                "BUREAU_BALANCE_SEVERE_DPD_RATIO",
            ]
        ],
        on=bureau_id,
        how="left",
    )
    return bureau

def add_pos_cash_balance(application_train, application_test):
    pos_cash_balance = pd.read_csv(RAW_DATA_PATH / "POS_CASH_balance.csv")
    numeric_cols = ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"]
    DPD_cols = ["SK_DPD", "SK_DPD_DEF"]
    agg_dict = {
        col: ["mean", "min", "max", "sum", "std"]
        for col in numeric_cols
    }
    agg_dict.update({
        col: ["mean", "max"] for col in DPD_cols
    })
    aggregated_pos_cash_balance = pos_cash_balance.groupby(ID_COL).agg(agg_dict)
    aggregated_pos_cash_balance.columns = [f"POS_CASH_{col}_{agg}".upper() for col, agg in aggregated_pos_cash_balance.columns]
    aggregated_pos_cash_balance = aggregated_pos_cash_balance.reset_index()
    pos_status_features = _make_status_features(
        pos_cash_balance,
        ID_COL,
        "NAME_CONTRACT_STATUS",
        "POS_CASH_CONTRACT_STATUS",
    )
    aggregated_pos_cash_balance = aggregated_pos_cash_balance.merge(
        pos_status_features,
        on=ID_COL,
        how="left",
    )
    application_train = application_train.merge(
        aggregated_pos_cash_balance,
        on=ID_COL,
        how="left",
    )
    application_test = application_test.merge(
        aggregated_pos_cash_balance,
        on=ID_COL,
        how="left",
    )
    return application_train, application_test

def add_previous_application(application_train, application_test):
    previous_application = pd.read_csv(RAW_DATA_PATH / "previous_application.csv")
    previous_application["APPLICATION_CREDIT_RATIO"] = _safe_divide(
        previous_application["AMT_APPLICATION"],
        previous_application["AMT_CREDIT"],
    )
    previous_application["GOODS_CREDIT_RATIO"] = _safe_divide(
        previous_application["AMT_GOODS_PRICE"],
        previous_application["AMT_CREDIT"],
    )
    previous_application["DOWN_PAYMENT_CREDIT_RATIO"] = _safe_divide(
        previous_application["AMT_DOWN_PAYMENT"],
        previous_application["AMT_CREDIT"],
    )
    numeric_cols = [
        col
        for col in previous_application.select_dtypes(include="number").columns
        if col not in [ID_COL, "SK_ID_PREV"]
    ]
    categorical_cols = previous_application.select_dtypes(exclude="number").columns.tolist()
    agg_dict = {
        col: ["mean", "min", "max", "sum", "std"]
        for col in numeric_cols
    }
    agg_dict.update({
        col: ["nunique"] for col in categorical_cols
    })
    aggregated_previous_application = previous_application.groupby(ID_COL).agg(agg_dict)
    aggregated_previous_application.columns = [f"PREVIOUS_{col}_{agg}".upper() for col, agg in aggregated_previous_application.columns]
    aggregated_previous_application = aggregated_previous_application.reset_index()
    previous_status_features = _make_status_features(
        previous_application,
        ID_COL,
        "NAME_CONTRACT_STATUS",
        "PREVIOUS_CONTRACT_STATUS",
    )
    aggregated_previous_application = aggregated_previous_application.merge(
        previous_status_features,
        on=ID_COL,
        how="left",
    )
    previous_counts = (
        previous_application.groupby(ID_COL)
        .size()
        .reset_index(name="PREVIOUS_RECORD_COUNT")
    )
    aggregated_previous_application = aggregated_previous_application.merge(
        previous_counts,
        on=ID_COL,
        how="left",
    )
    application_train = application_train.merge(
        aggregated_previous_application,
        on=ID_COL,
        how="left",
    )
    application_test = application_test.merge(
        aggregated_previous_application,
        on=ID_COL,
        how="left",
    )
    return application_train, application_test

def add_credit_card_balance(application_train, application_test):
    credit_card_balance = pd.read_csv(RAW_DATA_PATH / "credit_card_balance.csv")
    credit_card_balance["CREDIT_UTILIZATION"] = _safe_divide(
        credit_card_balance["AMT_BALANCE"],
        credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"],
    )
    credit_card_balance["DRAWING_RATIO"] = _safe_divide(
        credit_card_balance["AMT_DRAWINGS_CURRENT"],
        credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"],
    )
    credit_card_balance["PAYMENT_RATIO"] = _safe_divide(
        credit_card_balance["AMT_PAYMENT_CURRENT"],
        credit_card_balance["AMT_INST_MIN_REGULARITY"],
    )
    dpd_cols = ["SK_DPD", "SK_DPD_DEF"]
    numeric_cols = [c for c in credit_card_balance.columns if c not in dpd_cols + ["SK_ID_PREV", ID_COL, "NAME_CONTRACT_STATUS"]]
    agg_dict = {
        col: ["mean", "min", "max", "sum", "std"]
        for col in numeric_cols
    }
    agg_dict.update({
        col: ["mean", "max"] for col in dpd_cols
    })
    aggregated_credit_card_balance = credit_card_balance.groupby(ID_COL).agg(agg_dict)
    aggregated_credit_card_balance.columns = [f"CREDIT_CARD_{col}_{agg}".upper() for col, agg in aggregated_credit_card_balance.columns]
    aggregated_credit_card_balance = aggregated_credit_card_balance.reset_index()
    credit_card_status_features = _make_status_features(
        credit_card_balance,
        ID_COL,
        "NAME_CONTRACT_STATUS",
        "CREDIT_CARD_CONTRACT_STATUS",
    )
    aggregated_credit_card_balance = aggregated_credit_card_balance.merge(
        credit_card_status_features,
        on=ID_COL,
        how="left",
    )
    application_train = application_train.merge(
        aggregated_credit_card_balance,
        on=ID_COL,
        how="left",
    )
    application_test = application_test.merge(
        aggregated_credit_card_balance,
        on=ID_COL,
        how="left",
    )
    return application_train, application_test

def add_installments_payments(application_train, application_test):
    installments_payments = pd.read_csv(RAW_DATA_PATH / "installments_payments.csv")
    installments_payments["PAYMENT_DELAY"] = (
        installments_payments["DAYS_ENTRY_PAYMENT"] - installments_payments["DAYS_INSTALMENT"]
    )
    installments_payments["PAYMENT_DIFFERENCE"] = (
        installments_payments["AMT_INSTALMENT"] - installments_payments["AMT_PAYMENT"]
    )
    installments_payments["PAYMENT_LATE"] = (installments_payments["PAYMENT_DELAY"] > 0).astype(int)
    installments_payments["PAYMENT_SEVERE_LATE"] = (installments_payments["PAYMENT_DELAY"] > 30).astype(int)
    installments_payments["PAYMENT_UNDERPAID"] = (installments_payments["PAYMENT_DIFFERENCE"] > 0).astype(int)
    numeric_cols = [
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
        "PAYMENT_DELAY",
        "PAYMENT_DIFFERENCE",
    ]
    agg_dict = {
        col: ["mean", "min", "max", "sum", "std"]
        for col in numeric_cols
    }
    agg_dict.update({
        col: ["mean", "sum"]
        for col in ["PAYMENT_LATE", "PAYMENT_SEVERE_LATE", "PAYMENT_UNDERPAID"]
    })
    aggregated_installments_payments = installments_payments.groupby(ID_COL).agg(agg_dict)
    aggregated_installments_payments.columns = [f"INSTALLMENTS_{col}_{agg}".upper() for col, agg in aggregated_installments_payments.columns]
    aggregated_installments_payments = aggregated_installments_payments.reset_index()
    application_train = application_train.merge(
        aggregated_installments_payments,
        on=ID_COL,
        how="left",
    )
    application_test = application_test.merge(
        aggregated_installments_payments,
        on=ID_COL,
        how="left",
    )
    return application_train, application_test
