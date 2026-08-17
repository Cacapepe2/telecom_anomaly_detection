import pandas as pd

def diagnose(df):
    return pd.DataFrame({
        "data_type": df.dtypes,
        "null_count": df.isna().sum(),
        "pct_null": (df.isna().mean() * 100).round(2),
        "unique_count": df.nunique()
    })


def date_time(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    return df