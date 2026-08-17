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

PERCENTAGE_COLS = ["accessibility_pct","availability_pct","csfb_preparation_success_pct","erab_success_pct","handover_success_pct","volte_inter_freq_handover_success_pct","rrc_success_pct",
                   "s1_signaling_success_pct","volte_voice_completion_pct","session_drop_rate_pct","voice_drop_rate_pct","volte_handover_success_pct"]

VOLUMETRIC_COLS = ["volte_ul_volume_mb","ul_volume_mb","volte_dl_volume_mb","dl_volume_mb", "avg_volte_users","avg_users","avg_total_users","volte_voice_traffic","ul_throughput_kbps","dl_throughput_kbps" ]

OTHERS_COLS = ["pusch_interference_dbm","dl_buffer_latency_ms","volte_dl_buffer_latency_ms"]


def validate_non_negative(df,colluns):
    result = {}
    for col in colluns:
        result[col] = (df[col].dropna().lt(0).sum())
        return result