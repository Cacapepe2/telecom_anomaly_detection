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

# ============================================================
# GROUPING AND IDENTIFICATION COLUMNS
# ============================================================

SITE_INFO_COLS_4G = [
    "technology",
    "region_id",
    "site_id",
    "node_id",
]

SITE_INFO_COLS_5G = [
    "technology",
    "region_id",
    "site_id",
    "node_id",
]


# 4G is evaluated by site and sector
GROUP_COLS_4G = [
    "site_id",
    "sector_id",
]


# 5G is evaluated by site, sector and band
GROUP_COLS_5G = [
    "site_id",
    "sector_id",
    "band_category",
]


KEY_4G = [
    "timestamp",
    "site_id",
    "sector_id",
]


KEY_5G = [
    "timestamp",
    "site_id",
    "sector_id",
    "band_category",
]

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

# ============================================================
# PASS / NO PASS THRESHOLDS
# ============================================================

THRESHOLDS_4G = {

    "accessibility_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "availability_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "csfb_preparation_success_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "erab_success_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "handover_success_pct":
        {"op": "gt", "value": 97, "min_pass_ratio": 0.80},

    "volte_inter_freq_handover_success_pct":
        {"op": "gt", "value": 97, "min_pass_ratio": 0.80},

    "rrc_success_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "s1_signaling_success_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "pusch_interference_dbm":
        {"op": "lt", "value": -110, "min_pass_ratio": 0.80},

    "dl_buffer_latency_ms":
        {"op": "lt", "value": 15, "min_pass_ratio": 0.80},

    "volte_dl_buffer_latency_ms":
        {"op": "lt", "value": 15, "min_pass_ratio": 0.80},

    "volte_voice_completion_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    # 5 Mb = 5000 kbps
    "dl_throughput_kbps":
        {"op": "gt", "value": 5000, "min_pass_ratio": 0.80},

    "session_drop_rate_pct":
        {"op": "lt", "value": 0.4, "min_pass_ratio": 0.80},

    "voice_drop_rate_pct":
        {"op": "lt", "value": 0.4, "min_pass_ratio": 0.80},

    "volte_handover_success_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "volte_voice_traffic":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "avg_total_users":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "avg_users":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "avg_volte_users":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "dl_volume_mb":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "volte_dl_volume_mb":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "ul_volume_mb":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "volte_ul_volume_mb":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    # Reverse throughput requires only 70% of samples
    "ul_throughput_kbps":
        {"op": "gt", "value": 1000, "min_pass_ratio": 0.70},
}
def check_threshold(series, rule):

    if rule["op"] == "gt":
        return series > rule["value"]

    if rule["op"] == "lt":
        return series < rule["value"]

    raise ValueError(
        f"Unsupported operator: {rule['op']}"
    )
    
def evaluate_indicators(
    df,
    thresholds,
    group_cols,
    min_valid_ratio=0.80
):

    results = []

    for group_values, group in df.groupby(
        group_cols,
        observed=True,
        sort=False
    ):

        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_info = dict(
            zip(group_cols, group_values)
        )

        total_samples = len(group)

        for indicator, rule in thresholds.items():

            valid = group[indicator].dropna()

            valid_samples = len(valid)

            valid_ratio = (
                valid_samples / total_samples
                if total_samples > 0
                else 0
            )

            if valid_ratio < min_valid_ratio:

                status = "INSUFFICIENT_DATA"
                pass_ratio = None

            else:

                within_limit = check_threshold(
                    valid,
                    rule
                )

                pass_ratio = within_limit.mean()

                if pass_ratio >= rule["min_pass_ratio"]:
                    status = "PASS"
                else:
                    status = "NO_PASS"

            results.append({

                **group_info,

                "indicator": indicator,

                "threshold": rule["value"],

                "required_pass_ratio":
                    rule["min_pass_ratio"],

                "total_samples":
                    total_samples,

                "valid_samples":
                    valid_samples,

                "valid_ratio":
                    valid_ratio,

                "pass_ratio":
                    pass_ratio,

                "indicator_status":
                    status
            })

    return pd.DataFrame(results)

def summarize_site_indicators(indicator_results):

    def aggregate_status(status):

        if (status == "NO_PASS").any():
            return "NO_PASS"

        if (status == "INSUFFICIENT_DATA").any():
            return "INSUFFICIENT_DATA"

        return "PASS"

    return (
        indicator_results
        .groupby(
            ["site_id", "indicator"],
            observed=True
        )["indicator_status"]
        .agg(aggregate_status)
        .reset_index()
    )
    
def classify_sites(
    site_indicator_results,
    no_pass_ratio_threshold=0.10
):

    summary = (
        site_indicator_results
        .groupby(
            "site_id",
            observed=True
        )
        .agg(

            total_indicators=(
                "indicator",
                "nunique"
            ),

            no_pass_indicators=(
                "indicator_status",
                lambda x:
                    (x == "NO_PASS").sum()
            ),

            insufficient_indicators=(
                "indicator_status",
                lambda x:
                    (x == "INSUFFICIENT_DATA").sum()
            )
        )
        .reset_index()
    )

    summary["no_pass_indicator_ratio"] = (
        summary["no_pass_indicators"]
        / summary["total_indicators"]
    )

    summary["site_classification"] = "PASS"

    summary.loc[
        summary["insufficient_indicators"] > 0,
        "site_classification"
    ] = "INSUFFICIENT_DATA"

    summary.loc[
        summary["no_pass_indicator_ratio"] >= 0.10,
        "site_classification"
    ] = "NO_PASS"

    return summary

def calculate_quantile(
    df,
    group_cols,
    volumetric_cols,
    quantile
):

    return (
        df
        .groupby(
            group_cols,
            observed=True
        )[volumetric_cols]
        .quantile(quantile)
        .reset_index()
    )

THRESHOLDS_5G = {

    "sgnb_endc_drop_rate_pct":
        {"op": "lt", "value": 2, "min_pass_ratio": 0.80},

    "inter_sa_handover_success_pct":
        {"op": "gt", "value": 97, "min_pass_ratio": 0.80},

    "sa_rrc_success_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "sa_qos_flow_success_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "intra_sa_handover_success_pct":
        {"op": "gt", "value": 97, "min_pass_ratio": 0.80},

    "sa_data_accessibility_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "availability_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "avg_sa_users":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "ng_signaling_success_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "inter_nsa_handover_success_pct":
        {"op": "gt", "value": 97, "min_pass_ratio": 0.80},

    "intra_nsa_handover_success_pct":
        {"op": "gt", "value": 97, "min_pass_ratio": 0.80},

    # 100 Mb = 100000 kbps
    "dl_throughput_kbps":
        {"op": "gt", "value": 100000, "min_pass_ratio": 0.80},

    "nsa_endc_drop_rate_pct":
        {"op": "lt", "value": 2, "min_pass_ratio": 0.80},

    "avg_users":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "latency_ms":
        {"op": "lt", "value": 10, "min_pass_ratio": 0.80},

    "sa_drop_rate_pct":
        {"op": "lt", "value": 2, "min_pass_ratio": 0.80},

    "dl_volume_mb":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "nsa_data_accessibility_pct":
        {"op": "gt", "value": 99, "min_pass_ratio": 0.80},

    "ul_volume_mb":
        {"op": "gt", "value": 0, "min_pass_ratio": 0.80},

    "vonr_drop_rate_pct":
        {"op": "lt", "value": 2, "min_pass_ratio": 0.80},

    "ul_interference_dbm":
        {"op": "lt", "value": -110, "min_pass_ratio": 0.80},

    # 50 Mb = 50000 kbps
    "ul_throughput_kbps":
        {"op": "gt", "value": 50000, "min_pass_ratio": 0.70},
}