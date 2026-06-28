from src.data_loading import RAW_DATA_PATH, ID_COL, load_data
import pandas as pd

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
