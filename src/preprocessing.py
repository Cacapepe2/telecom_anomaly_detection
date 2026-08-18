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

PERCENTAGE_COLS_4G = ["accessibility_pct","availability_pct","csfb_preparation_success_pct","erab_success_pct","handover_success_pct","volte_inter_freq_handover_success_pct","rrc_success_pct",
                   "s1_signaling_success_pct","volte_voice_completion_pct","session_drop_rate_pct","voice_drop_rate_pct","volte_handover_success_pct"]

VOLUMETRIC_COLS_4G = ["volte_ul_volume_mb","ul_volume_mb","volte_dl_volume_mb","dl_volume_mb", "avg_volte_users","avg_users","avg_total_users","volte_voice_traffic","ul_throughput_kbps","dl_throughput_kbps" ]

OTHERS_COLS_4G = ["pusch_interference_dbm","dl_buffer_latency_ms","volte_dl_buffer_latency_ms"]

PERCENTAGE_COLS_5G =["sgnb_endc_drop_rate_pct","inter_sa_handover_success_pct","sa_rrc_success_pct","sa_qos_flow_success_pct","intra_sa_handover_success_pct",
                     "sa_data_accessibility_pct","ng_signaling_success_pct","inter_nsa_handover_success_pct","intra_nsa_handover_success_pct","nsa_endc_drop_rate_pct",
                     "sa_drop_rate_pct","nsa_data_accessibility_pct","vonr_drop_rate_pct"]

VOLUMETRIC_COLS_5G = ["ul_throughput_kbps","avg_sa_users","dl_throughput_kbps","avg_users","dl_volume_mb","ul_volume_mb"]

OTHERS_COLS_5G =["latency_ms","ul_interference_dbm"]

def validate_non_negative(df,colluns):
    result = {}
    for col in colluns:
        result[col] = (df[col].dropna().lt(0).sum())
    return result

def remove_invalid_rows(df, volumetric_cols, percentage_cols):
    df = df.copy()

    invalid_volumetric = (
        df[volumetric_cols]
        .lt(0)
        .any(axis=1)
    )

    invalid_percentage = (
        (
            df[percentage_cols].lt(0)
            | df[percentage_cols].gt(100)
        )
        .any(axis=1)
    )

    invalid_rows = invalid_volumetric | invalid_percentage

    rejected = df[invalid_rows].copy()

    rejected["rejection_reason"] = "invalid_capture"

    clean = df[~invalid_rows].copy()

    return clean, rejected