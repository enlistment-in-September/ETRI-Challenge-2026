import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import RAW_DIR
from src.utils import safe_mean, safe_std, safe_sum, safe_max, safe_entropy


def _pfile(name):
    return str(RAW_DIR / name)


def _safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")


def ensure_datetime_cols(df, ts_col="timestamp"):
    df = df.copy()
    df[ts_col] = _safe_to_datetime(df[ts_col])
    df = df[df[ts_col].notna()].copy()
    df["lifelog_date"] = df[ts_col].dt.date.astype(str)
    df["hour"] = df[ts_col].dt.hour
    return df


def agg_num_features(df, value_col, prefix):
    if value_col not in df.columns:
        return None
    return (
        df.groupby(["subject_id", "lifelog_date"], as_index=False)
          .agg(**{
              f"{prefix}_mean": (value_col, "mean"),
              f"{prefix}_std": (value_col, "std"),
              f"{prefix}_min": (value_col, "min"),
              f"{prefix}_max": (value_col, "max"),
              f"{prefix}_median": (value_col, "median"),
              f"{prefix}_count": (value_col, "count"),
          })
    )


def agg_hour_bins(df, value_col, prefix):
    if value_col not in df.columns:
        return None
    bins = {
        "night": [0, 1, 2, 3, 4, 5],
        "morning": [6, 7, 8, 9, 10, 11],
        "afternoon": [12, 13, 14, 15, 16, 17],
        "evening": [18, 19, 20, 21, 22, 23],
    }
    out = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False)
          .agg(**{f"{prefix}_daily_mean": (value_col, "mean")})
    )
    for name, hours in bins.items():
        tmp = (
            df[df["hour"].isin(hours)]
            .groupby(["subject_id", "lifelog_date"], as_index=False)
            .agg(**{f"{prefix}_{name}_mean": (value_col, "mean")})
        )
        out = out.merge(tmp, on=["subject_id", "lifelog_date"], how="left")
    for name in bins:
        out[f"{prefix}_{name}_ratio"] = out[f"{prefix}_{name}_mean"] / (out[f"{prefix}_daily_mean"] + 1e-6)
    return out


def merge_feature_list(base_df, feat_list):
    out = base_df.copy()
    for i, feat in enumerate(feat_list):
        if feat is None or len(feat) == 0:
            continue
        out = out.merge(feat, on=["subject_id", "lifelog_date"], how="left")
        print(f"[MERGE] {i+1}/{len(feat_list)} -> {out.shape}")
    return out


def add_calendar_feats(df):
    df = df.copy()
    dt = pd.to_datetime(df["lifelog_date"])
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_stats(df, exclude_cols):
    df = df.copy()
    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols:
        g = df.groupby("subject_id")[col]
        subj_mean = g.transform("mean")
        subj_std = g.transform("std")
        df[f"{col}_subj_mean"] = subj_mean
        df[f"{col}_subj_std"] = subj_std
        df[f"{col}_subj_z"] = (df[col] - subj_mean) / (subj_std + 1e-6)
    return df


def add_recent_stats(df, exclude_cols, max_cols=60):
    df = df.copy().sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)
    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_cols if df[c].nunique(dropna=True) > 1][:max_cols]
    for col in num_cols:
        g = df.groupby("subject_id")[col]
        shifted = g.shift(1)
        df[f"{col}_lag1"] = shifted
        df[f"{col}_lag2"] = g.shift(2)
        df[f"{col}_diff1"] = df[col] - shifted
        df[f"{col}_roll3_mean"] = (
            shifted.groupby(df["subject_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{col}_roll7_mean"] = (
            shifted.groupby(df["subject_id"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        )
    return df


def sanitize_numeric_df(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def build_lstm_features(base_all):
    """B.py???쇱쿂 異붿텧 ?꾩껜 ?뚯씠?꾨씪?? base_all = (subject_id, lifelog_date) 湲곗? ?곗씠?고봽?덉엫."""
    feature_tables = []

    if (_pfile("ch2025_mLight.parquet") and (RAW_DIR / "ch2025_mLight.parquet").exists()):
        df = pd.read_parquet(_pfile("ch2025_mLight.parquet"))
        df = ensure_datetime_cols(df)
        feature_tables += [agg_num_features(df, "m_light", "m_light"), agg_hour_bins(df, "m_light", "m_light")]
        del df; gc.collect()

    if (RAW_DIR / "ch2025_wLight.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_wLight.parquet"))
        df = ensure_datetime_cols(df)
        feature_tables += [agg_num_features(df, "w_light", "w_light"), agg_hour_bins(df, "w_light", "w_light")]
        del df; gc.collect()

    if (RAW_DIR / "ch2025_mACStatus.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_mACStatus.parquet"))
        df = ensure_datetime_cols(df)
        if "m_charging" in df.columns:
            feat1 = df.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                m_charging_mean=("m_charging", "mean"),
                m_charging_sum=("m_charging", "sum"),
                m_charging_std=("m_charging", "std"),
                m_charging_count=("m_charging", "count"),
            )
            feature_tables += [feat1, agg_hour_bins(df, "m_charging", "m_charging")]
        del df; gc.collect()

    if (RAW_DIR / "ch2025_mScreenStatus.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_mScreenStatus.parquet"))
        df = ensure_datetime_cols(df)
        if "m_screen_use" in df.columns:
            feat1 = df.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                m_screen_use_mean=("m_screen_use", "mean"),
                m_screen_use_sum=("m_screen_use", "sum"),
                m_screen_use_std=("m_screen_use", "std"),
                m_screen_use_count=("m_screen_use", "count"),
            )
            feature_tables += [feat1, agg_hour_bins(df, "m_screen_use", "m_screen_use")]
        del df; gc.collect()

    if (RAW_DIR / "ch2025_mActivity.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_mActivity.parquet"))
        df = ensure_datetime_cols(df)
        if "m_activity" in df.columns:
            feat_num = df.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                m_activity_mean=("m_activity", "mean"),
                m_activity_std=("m_activity", "std"),
                m_activity_min=("m_activity", "min"),
                m_activity_max=("m_activity", "max"),
                m_activity_median=("m_activity", "median"),
                m_activity_nunique=("m_activity", "nunique"),
                m_activity_count=("m_activity", "count"),
            )
            act_counts = df.groupby(["subject_id", "lifelog_date", "m_activity"]).size().reset_index(name="cnt")
            pivot = act_counts.pivot_table(
                index=["subject_id", "lifelog_date"], columns="m_activity", values="cnt", fill_value=0
            )
            pivot.columns = [f"m_activity_code_{c}_cnt" for c in pivot.columns]
            feature_tables += [feat_num, pivot.reset_index()]
        del df; gc.collect()

    if (RAW_DIR / "ch2025_wPedo.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_wPedo.parquet"))
        df = ensure_datetime_cols(df)
        pedo_cols = ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"]
        use_cols = [c for c in pedo_cols if c in df.columns]
        if use_cols:
            agg_dict = {}
            for c in use_cols:
                for stat in ["mean", "sum", "std", "max", "median"]:
                    agg_dict[f"{c}_{stat}"] = (c, stat)
            feature_tables.append(df.groupby(["subject_id", "lifelog_date"], as_index=False).agg(**agg_dict))
        del df; gc.collect()

    if (RAW_DIR / "ch2025_wHr.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_wHr.parquet"))
        df = ensure_datetime_cols(df)
        if "heart_rate" in df.columns:
            rows = []
            for r in df.itertuples(index=False):
                hr = getattr(r, "heart_rate", None)
                try:
                    arr = np.asarray(hr, dtype=float) if isinstance(hr, (list, tuple, np.ndarray)) else np.asarray([hr], dtype=float)
                    arr = arr[np.isfinite(arr)]
                except Exception:
                    arr = np.array([])
                if len(arr) == 0:
                    rows.append([r.subject_id, r.lifelog_date, np.nan, np.nan, np.nan, np.nan, 0])
                else:
                    rows.append([r.subject_id, r.lifelog_date, arr.mean(), arr.std(), arr.min(), arr.max(), len(arr)])
            hr_df = pd.DataFrame(rows, columns=["subject_id", "lifelog_date", "hr_mean", "hr_std", "hr_min", "hr_max", "hr_len"])
            feature_tables.append(
                hr_df.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                    hr_mean_mean=("hr_mean", "mean"), hr_mean_std=("hr_mean", "std"),
                    hr_std_mean=("hr_std", "mean"), hr_std_max=("hr_std", "max"),
                    hr_min_mean=("hr_min", "mean"), hr_max_mean=("hr_max", "mean"),
                    hr_len_sum=("hr_len", "sum"), hr_len_mean=("hr_len", "mean"),
                )
            )
        del df; gc.collect()

    if (RAW_DIR / "ch2025_mUsageStats.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_mUsageStats.parquet"))
        df = ensure_datetime_cols(df)
        if "m_usage_stats" in df.columns:
            rows = []
            for r in df.itertuples(index=False):
                items = getattr(r, "m_usage_stats", None)
                if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                    rows.append([r.subject_id, r.lifelog_date, 0, 0, 0.0, 0.0])
                    continue
                total_times, app_names = [], []
                for item in items:
                    if isinstance(item, dict):
                        app_names.append(str(item.get("app_name", "UNK")))
                        try:
                            total_times.append(float(item.get("total_time", 0)))
                        except Exception:
                            total_times.append(0.0)
                rows.append([r.subject_id, r.lifelog_date, len(items), len(set(app_names)), safe_sum(total_times), safe_entropy(total_times)])
            feat = pd.DataFrame(rows, columns=["subject_id", "lifelog_date", "usage_event_app_cnt", "usage_event_app_unique", "usage_event_total_time", "usage_event_time_entropy"])
            feature_tables.append(
                feat.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                    usage_event_app_cnt_mean=("usage_event_app_cnt", "mean"),
                    usage_event_app_cnt_sum=("usage_event_app_cnt", "sum"),
                    usage_event_app_unique_mean=("usage_event_app_unique", "mean"),
                    usage_event_app_unique_sum=("usage_event_app_unique", "sum"),
                    usage_event_total_time_mean=("usage_event_total_time", "mean"),
                    usage_event_total_time_sum=("usage_event_total_time", "sum"),
                    usage_event_time_entropy_mean=("usage_event_time_entropy", "mean"),
                    usage_event_time_entropy_max=("usage_event_time_entropy", "max"),
                )
            )
        del df; gc.collect()

    if (RAW_DIR / "ch2025_mWifi.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_mWifi.parquet"))
        df = ensure_datetime_cols(df)
        if "m_wifi" in df.columns:
            rows = []
            for r in df.itertuples(index=False):
                items = getattr(r, "m_wifi", None)
                if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                    rows.append([r.subject_id, r.lifelog_date, 0, 0, np.nan, np.nan])
                    continue
                rssis, bssids = [], []
                for item in items:
                    if isinstance(item, dict):
                        bssids.append(str(item.get("bssid", "UNK")))
                        try:
                            rssis.append(float(item.get("rssi", np.nan)))
                        except Exception:
                            pass
                rows.append([r.subject_id, r.lifelog_date, len(items), len(set(bssids)), safe_mean(rssis), safe_max(rssis)])
            feat = pd.DataFrame(rows, columns=["subject_id", "lifelog_date", "wifi_scan_cnt", "wifi_unique_bssid", "wifi_rssi_mean", "wifi_rssi_max"])
            feature_tables.append(
                feat.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                    wifi_scan_cnt_mean=("wifi_scan_cnt", "mean"), wifi_scan_cnt_sum=("wifi_scan_cnt", "sum"),
                    wifi_unique_bssid_mean=("wifi_unique_bssid", "mean"), wifi_unique_bssid_sum=("wifi_unique_bssid", "sum"),
                    wifi_rssi_mean_mean=("wifi_rssi_mean", "mean"), wifi_rssi_mean_std=("wifi_rssi_mean", "std"),
                    wifi_rssi_max_mean=("wifi_rssi_max", "mean"), wifi_rssi_max_max=("wifi_rssi_max", "max"),
                )
            )
        del df; gc.collect()

    if (RAW_DIR / "ch2025_mBle.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_mBle.parquet"))
        df = ensure_datetime_cols(df)
        if "m_ble" in df.columns:
            rows = []
            for r in df.itertuples(index=False):
                items = getattr(r, "m_ble", None)
                if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                    rows.append([r.subject_id, r.lifelog_date, 0, 0, np.nan, 0])
                    continue
                rssis, addrs, classes = [], [], []
                for item in items:
                    if isinstance(item, dict):
                        addrs.append(str(item.get("address", "UNK")))
                        classes.append(str(item.get("device_class", "UNK")))
                        try:
                            rssis.append(float(item.get("rssi", np.nan)))
                        except Exception:
                            pass
                rows.append([r.subject_id, r.lifelog_date, len(items), len(set(addrs)), safe_mean(rssis), len(set(classes))])
            feat = pd.DataFrame(rows, columns=["subject_id", "lifelog_date", "ble_scan_cnt", "ble_unique_addr", "ble_rssi_mean", "ble_device_class_unique"])
            feature_tables.append(
                feat.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                    ble_scan_cnt_mean=("ble_scan_cnt", "mean"), ble_scan_cnt_sum=("ble_scan_cnt", "sum"),
                    ble_unique_addr_mean=("ble_unique_addr", "mean"), ble_unique_addr_sum=("ble_unique_addr", "sum"),
                    ble_rssi_mean_mean=("ble_rssi_mean", "mean"), ble_rssi_mean_std=("ble_rssi_mean", "std"),
                    ble_device_class_unique_mean=("ble_device_class_unique", "mean"),
                    ble_device_class_unique_sum=("ble_device_class_unique", "sum"),
                )
            )
        del df; gc.collect()

    if (RAW_DIR / "ch2025_mGps.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_mGps.parquet"))
        df = ensure_datetime_cols(df)
        if "m_gps" in df.columns:
            rows = []
            for r in df.itertuples(index=False):
                items = getattr(r, "m_gps", None)
                if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                    rows.append([r.subject_id, r.lifelog_date, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    continue
                lats, lons, speeds = [], [], []
                for item in items:
                    if isinstance(item, dict):
                        try: lats.append(float(item.get("latitude", np.nan)))
                        except Exception: pass
                        try: lons.append(float(item.get("longitude", np.nan)))
                        except Exception: pass
                        try: speeds.append(float(item.get("speed", np.nan)))
                        except Exception: pass
                lat_std = safe_std(lats)
                lon_std = safe_std(lons)
                spread = np.sqrt(lat_std**2 + lon_std**2) if np.isfinite(lat_std) and np.isfinite(lon_std) else np.nan
                rows.append([r.subject_id, r.lifelog_date, len(items), safe_mean(lats), safe_mean(lons), lat_std, lon_std, safe_mean(speeds), spread])
            feat = pd.DataFrame(rows, columns=["subject_id", "lifelog_date", "gps_point_cnt", "gps_lat_mean", "gps_lon_mean", "gps_lat_std", "gps_lon_std", "gps_speed_mean", "gps_spatial_spread"])
            feature_tables.append(
                feat.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                    gps_point_cnt_mean=("gps_point_cnt", "mean"), gps_point_cnt_sum=("gps_point_cnt", "sum"),
                    gps_lat_mean_mean=("gps_lat_mean", "mean"), gps_lon_mean_mean=("gps_lon_mean", "mean"),
                    gps_lat_std_mean=("gps_lat_std", "mean"), gps_lon_std_mean=("gps_lon_std", "mean"),
                    gps_speed_mean_mean=("gps_speed_mean", "mean"), gps_speed_mean_max=("gps_speed_mean", "max"),
                    gps_spatial_spread_mean=("gps_spatial_spread", "mean"), gps_spatial_spread_max=("gps_spatial_spread", "max"),
                )
            )
        del df; gc.collect()

    if (RAW_DIR / "ch2025_mAmbience.parquet").exists():
        df = pd.read_parquet(_pfile("ch2025_mAmbience.parquet"))
        df = ensure_datetime_cols(df)
        if "m_ambience" in df.columns:
            rows = []
            for r in df.itertuples(index=False):
                items = getattr(r, "m_ambience", None)
                if items is None or not isinstance(items, (list, tuple)) or len(items) == 0:
                    rows.append([r.subject_id, r.lifelog_date, 0, 0, 0.0, 0.0, 0.0, 0.0])
                    continue
                labels_list = []
                music_score = speech_score = vehicle_score = outside_score = 0.0
                for item in items:
                    label, score = None, 0.0
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        label = str(item[0])
                        try: score = float(item[1])
                        except Exception: score = 0.0
                    elif isinstance(item, dict):
                        label = str(item.get("label", "UNK"))
                        try: score = float(item.get("score", 0.0))
                        except Exception: score = 0.0
                    if label is None:
                        continue
                    labels_list.append(label)
                    if "Music" in label: music_score += score
                    if "Speech" in label: speech_score += score
                    if any(x in label for x in ["Vehicle", "Car", "Truck"]): vehicle_score += score
                    if "Outside" in label: outside_score += score
                rows.append([r.subject_id, r.lifelog_date, len(labels_list), len(set(labels_list)), music_score, speech_score, vehicle_score, outside_score])
            feat = pd.DataFrame(rows, columns=["subject_id", "lifelog_date", "amb_label_cnt", "amb_label_unique", "amb_music_score", "amb_speech_score", "amb_vehicle_score", "amb_outside_score"])
            feature_tables.append(
                feat.groupby(["subject_id", "lifelog_date"], as_index=False).agg(
                    amb_label_cnt_mean=("amb_label_cnt", "mean"), amb_label_cnt_sum=("amb_label_cnt", "sum"),
                    amb_label_unique_mean=("amb_label_unique", "mean"), amb_label_unique_sum=("amb_label_unique", "sum"),
                    amb_music_score_mean=("amb_music_score", "mean"), amb_music_score_sum=("amb_music_score", "sum"),
                    amb_speech_score_mean=("amb_speech_score", "mean"), amb_speech_score_sum=("amb_speech_score", "sum"),
                    amb_vehicle_score_mean=("amb_vehicle_score", "mean"), amb_vehicle_score_sum=("amb_vehicle_score", "sum"),
                    amb_outside_score_mean=("amb_outside_score", "mean"), amb_outside_score_sum=("amb_outside_score", "sum"),
                )
            )
        del df; gc.collect()

    feature_tables = [x for x in feature_tables if x is not None and len(x) > 0]
    print("feature_tables:", len(feature_tables))
    return feature_tables


def select_stable_features(train_part, cols, max_features=256, missing_th=0.95):
    tmp = train_part[cols].copy().replace([np.inf, -np.inf], np.nan)
    miss = tmp.isna().mean()
    valid_cols = miss[miss < missing_th].index.tolist()
    if not valid_cols:
        return cols[:min(len(cols), max_features)]
    var_series = tmp[valid_cols].fillna(tmp[valid_cols].median()).var()
    return var_series.sort_values(ascending=False).index.tolist()[:max_features]


def fit_feature_stats(daily_part, cols):
    x = daily_part[cols].copy().replace([np.inf, -np.inf], np.nan)
    med = x.median()
    x = x.fillna(med).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(x)
    return med, scaler


def transform_features(df, cols, med, scaler):
    x = df[cols].copy().replace([np.inf, -np.inf], np.nan)
    x = x.fillna(med).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    arr = np.nan_to_num(scaler.transform(x), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(arr, columns=cols, index=df.index)


def build_subject_daily_arrays(daily_df, feature_cols, med, scaler):
    tmp = daily_df[["subject_id", "lifelog_date"] + feature_cols].copy()
    tmp["date_dt"] = pd.to_datetime(tmp["lifelog_date"])
    tmp_scaled = transform_features(tmp, feature_cols, med, scaler)
    tmp2 = pd.concat([
        tmp[["subject_id", "lifelog_date", "date_dt"]].reset_index(drop=True),
        tmp_scaled.reset_index(drop=True),
    ], axis=1)
    out = {}
    for sid, sdf in tmp2.groupby("subject_id"):
        sdf = sdf.sort_values("date_dt").reset_index(drop=True)
        feat = np.nan_to_num(sdf[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out[sid] = {"dates": sdf["lifelog_date"].tolist(), "date_dt": sdf["date_dt"].tolist(), "feat": feat}
    return out

