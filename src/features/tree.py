import numpy as np
import pandas as pd

from config import RAW_DIR
from src.utils import agg_stats, safe_mean, parse_hr


def load_parquet(name):
    df = pd.read_parquet(RAW_DIR / f'ch2025_{name}.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def extract_activity(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        acts = grp['m_activity'].values
        h = grp['timestamp'].dt.hour.values
        for a in [0, 3, 4, 7, 8]:
            row[f'act_{a}_ratio'] = (acts == a).mean()
        row['act_active_ratio'] = ((acts == 7) | (acts == 8) | (acts == 3)).mean()
        row['act_still_ratio'] = (acts == 0).mean()
        row['act_n_records'] = len(acts)
        for seg, mask in [('morn', (h >= 6) & (h < 12)), ('aftn', (h >= 12) & (h < 18)),
                          ('eve', (h >= 18) & (h < 22)), ('night', (h >= 22) | (h < 6))]:
            s_acts = acts[mask]
            row[f'act_{seg}_active'] = ((s_acts == 7) | (s_acts == 8)).mean() if len(s_acts) > 0 else np.nan
            row[f'act_{seg}_still'] = (s_acts == 0).mean() if len(s_acts) > 0 else np.nan
        pre = acts[(h >= 22) & (h < 24)]
        row['act_presleep_active'] = ((pre == 7) | (pre == 8)).mean() if len(pre) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_pedo(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row['pedo_total_steps'] = grp['step'].sum()
        row['pedo_total_distance'] = grp['distance'].sum()
        row['pedo_total_calories'] = grp['burned_calories'].sum()
        row['pedo_max_speed'] = grp['speed'].max()
        row['pedo_mean_speed'] = grp['speed'].mean()
        row['pedo_running_steps'] = grp['running_step'].sum()
        row['pedo_walking_steps'] = grp['walking_step'].sum()
        row['pedo_run_ratio'] = grp['running_step'].sum() / (grp['step'].sum() + 1)
        eve = grp[grp['timestamp'].dt.hour.between(18, 21)]
        row['pedo_evening_steps'] = eve['step'].sum()
        row['pedo_step_freq_mean'] = grp['step_frequency'].mean()
        row['pedo_step_freq_max'] = grp['step_frequency'].max()
        hourly = grp.groupby(grp['timestamp'].dt.hour)['step'].sum()
        row['pedo_active_hours'] = (hourly > 50).sum()
        feats.append(row)
    return pd.DataFrame(feats)


def extract_hr(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        all_v, seg_v = [], {'morn': [], 'aftn': [], 'eve': [], 'night': []}
        for ts, v in zip(grp['timestamp'], grp['heart_rate']):
            arr = parse_hr(v)
            all_v.extend(arr.tolist())
            h = ts.hour
            if 6 <= h < 12:
                seg_v['morn'].extend(arr.tolist())
            elif 12 <= h < 18:
                seg_v['aftn'].extend(arr.tolist())
            elif 18 <= h < 22:
                seg_v['eve'].extend(arr.tolist())
            else:
                seg_v['night'].extend(arr.tolist())
        hr = np.array(all_v)
        for k, v in agg_stats(hr, 'hr').items():
            row[k] = v
        row['hr_resting'] = float(np.nanpercentile(hr, 10)) if len(hr) > 0 else np.nan
        row['hr_active'] = float(np.nanpercentile(hr, 90)) if len(hr) > 0 else np.nan
        row['hr_rmssd'] = float(np.sqrt(np.nanmean(np.diff(hr) ** 2))) if len(hr) > 1 else np.nan
        row['hr_n_records'] = len(hr)
        for seg, vals in seg_v.items():
            a = np.array(vals)
            row[f'hr_{seg}_mean'] = float(np.nanmean(a)) if len(a) > 0 else np.nan
            row[f'hr_{seg}_std'] = float(np.nanstd(a)) if len(a) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_screen(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        sc = grp['m_screen_use'].values
        h = grp['timestamp'].dt.hour.values
        row['screen_on_total'] = (sc > 0).sum()
        row['screen_on_ratio'] = (sc > 0).mean()
        row['screen_unlock_cnt'] = (sc[1:] > sc[:-1]).sum()
        for seg, mask in [('night', (h >= 22) | (h < 2)), ('eve', (h >= 20) & (h <= 23)),
                          ('presleep', (h >= 22) & (h < 24))]:
            s_sc = sc[mask]
            row[f'screen_{seg}_on'] = (s_sc > 0).sum()
            row[f'screen_{seg}_ratio'] = (s_sc > 0).mean() if len(s_sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_light(df_raw, col, prefix, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        vals = grp[col].dropna().values
        for k, v in agg_stats(vals, f'{prefix}_all').items():
            row[k] = v
        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('eve', (18, 22)), ('morn', (6, 10)), ('night', (22, 24))]:
            sv = grp.loc[h.between(lo, hi - 1), col].dropna().values
            row[f'{prefix}_{seg}_mean'] = safe_mean(sv)
        row[f'{prefix}_dark_ratio'] = (vals < 10).mean() if len(vals) > 0 else np.nan
        row[f'{prefix}_bright_ratio'] = (vals > 1000).mean() if len(vals) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_ac(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        ch = grp['m_charging'].values
        h = grp['timestamp'].dt.hour.values
        row['ac_charging_ratio'] = ch.mean()
        for seg, mask in [('eve', (h >= 21) & (h <= 23)), ('night', (h >= 22) | (h < 4)),
                          ('presleep', (h >= 22) & (h < 24))]:
            sc = ch[mask]
            row[f'ac_{seg}_charging'] = sc.mean() if len(sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_gps(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        speeds, lats, lons = [], [], []
        for v in grp['m_gps']:
            if isinstance(v, list):
                for pt in v:
                    if isinstance(pt, dict):
                        speeds.append(pt.get('speed', 0))
                        lats.append(pt.get('latitude', 0))
                        lons.append(pt.get('longitude', 0))
        speeds = np.array(speeds)
        row['gps_mean_speed'] = np.nanmean(speeds) if len(speeds) > 0 else np.nan
        row['gps_max_speed'] = np.nanmax(speeds) if len(speeds) > 0 else np.nan
        row['gps_moving_ratio'] = (speeds > 0.5).mean() if len(speeds) > 0 else np.nan
        row['gps_lat_std'] = np.nanstd(lats) if len(lats) > 0 else np.nan
        row['gps_lon_std'] = np.nanstd(lons) if len(lons) > 0 else np.nan
        if len(lats) > 1:
            dlat, dlon = np.diff(lats), np.diff(lons)
            row['gps_total_disp'] = float(np.sum(np.sqrt(dlat ** 2 + dlon ** 2)))
        else:
            row['gps_total_disp'] = 0.0
        feats.append(row)
    return pd.DataFrame(feats)


def extract_usage(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        total_time, late_time, eve_time, n_apps = 0, 0, 0, 0
        for ts, v in zip(grp['timestamp'], grp['m_usage_stats']):
            if isinstance(v, list):
                for app in v:
                    if isinstance(app, dict):
                        t = app.get('total_time', 0) or 0
                        total_time += t
                        n_apps += 1
                        if ts.hour >= 22 or ts.hour < 2:
                            late_time += t
                        if ts.hour >= 18:
                            eve_time += t
        row['usage_total_time'] = total_time
        row['usage_n_apps'] = n_apps
        row['usage_late_time'] = late_time
        row['usage_late_ratio'] = late_time / (total_time + 1)
        row['usage_eve_time'] = eve_time
        row['usage_eve_ratio'] = eve_time / (total_time + 1)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_wifi(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        all_bssids, rssi_vals = set(), []
        for v in grp['m_wifi']:
            if isinstance(v, list):
                for net in v:
                    if isinstance(net, dict):
                        all_bssids.add(net.get('bssid', ''))
                        rssi_vals.append(net.get('rssi', -100))
        row['wifi_n_unique'] = len(all_bssids)
        row['wifi_mean_rssi'] = np.mean(rssi_vals) if rssi_vals else np.nan
        row['wifi_max_rssi'] = np.max(rssi_vals) if rssi_vals else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_ble(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        addrs = set()
        for v in grp['m_ble']:
            if isinstance(v, list):
                for dev in v:
                    if isinstance(dev, dict):
                        addrs.add(dev.get('address', ''))
        row['ble_n_unique'] = len(addrs)
        row['ble_n_scans'] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_wlight(df_raw, keys):
    return extract_light(df_raw, 'w_light', 'wlight', keys)


def extract_ambience(df_raw, keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        music_s, speech_s, silence_s = [], [], []
        for v in grp['m_ambience']:
            if isinstance(v, list):
                d_map = {item[0]: item[1] for item in v if isinstance(item, list) and len(item) == 2}
                music_s.append(d_map.get('Music', 0))
                speech_s.append(d_map.get('Speech', 0))
                silence_s.append(d_map.get('Silence', 0))
        row['amb_music_mean'] = np.mean(music_s) if music_s else np.nan
        row['amb_speech_mean'] = np.mean(speech_s) if speech_s else np.nan
        row['amb_silence_mean'] = np.mean(silence_s) if silence_s else np.nan
        row['amb_n_records'] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)


# ── 수면 시간대 피처 (sleep_date 기준) ──────────────────────────────────────

def extract_sleep_hr(df_raw, sleep_keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    df_m = df_raw[df_raw['timestamp'].dt.hour < 9].copy()
    feats = []
    for (sid, d), grp in df_m.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        hour_vals = {h: [] for h in range(9)}
        all_v = []
        for ts, v in zip(grp['timestamp'], grp['heart_rate']):
            arr = parse_hr(v)
            all_v.extend(arr.tolist())
            hour_vals[ts.hour].extend(arr.tolist())
        sleep_hrs = np.array(all_v)
        for k, v in agg_stats(sleep_hrs, 'slp_hr').items():
            row[k] = v
        row['slp_hr_deep_ratio'] = (sleep_hrs < 55).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_awake_ratio'] = (sleep_hrs > 75).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_light_ratio'] = ((sleep_hrs >= 55) & (sleep_hrs <= 75)).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_rmssd'] = float(np.sqrt(np.nanmean(np.diff(sleep_hrs) ** 2))) if len(sleep_hrs) > 1 else np.nan
        row['slp_hr_n_records'] = len(grp)
        row['slp_hr_early_mean'] = safe_mean(sum([hour_vals[h] for h in range(3)], []))
        row['slp_hr_late_mean'] = safe_mean(sum([hour_vals[h] for h in range(6, 9)], []))
        row['slp_hr_mid_mean'] = safe_mean(sum([hour_vals[h] for h in range(3, 6)], []))
        row['slp_hr_range'] = float(np.ptp(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_median'] = float(np.median(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 5:
            rolling = pd.Series(sleep_hrs).rolling(5, min_periods=1).mean().values
            row['slp_hr_spike_count'] = int((np.abs(sleep_hrs - rolling) > 15).sum())
        else:
            row['slp_hr_spike_count'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_pedo(df_raw, sleep_keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        row['slp_pedo_steps'] = morn['step'].sum()
        row['slp_pedo_active'] = (morn['step'] > 5).sum()
        row['slp_pedo_calories'] = morn['burned_calories'].sum()
        row['slp_pedo_n_records'] = len(morn)
        mid = grp[grp['timestamp'].dt.hour.between(2, 4)]
        row['slp_pedo_mid_steps'] = mid['step'].sum()
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_activity(df_raw, sleep_keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) == 0:
            row.update({'slp_act_still_ratio': np.nan, 'slp_act_active_ratio': np.nan, 'slp_act_n_records': 0})
        else:
            acts = morn['m_activity'].values
            row['slp_act_still_ratio'] = (acts == 0).mean()
            row['slp_act_active_ratio'] = ((acts == 7) | (acts == 8)).mean()
            row['slp_act_n_records'] = len(acts)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_screen(df_raw, sleep_keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            sc = morn['m_screen_use'].values
            row['slp_screen_on'] = (sc > 0).sum()
            row['slp_screen_ratio'] = (sc > 0).mean()
        else:
            row['slp_screen_on'] = row['slp_screen_ratio'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_light(df_raw, sleep_keys):
    df_raw = df_raw.copy()
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            vals = morn['w_light'].dropna().values
            row['slp_wlight_mean'] = safe_mean(vals)
            row['slp_wlight_dark'] = (vals < 5).mean() if len(vals) > 0 else np.nan
            row['slp_wlight_light'] = (vals > 100).mean() if len(vals) > 0 else np.nan
        else:
            row['slp_wlight_mean'] = row['slp_wlight_dark'] = row['slp_wlight_light'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def build_tree_features(train_df, test_df):
    import holidays

    all_keys = pd.concat([
        train_df[['subject_id', 'lifelog_date']],
        test_df[['subject_id', 'lifelog_date']],
    ]).drop_duplicates().reset_index(drop=True)

    sleep_keys = pd.concat([
        train_df[['subject_id', 'sleep_date']],
        test_df[['subject_id', 'sleep_date']],
    ]).drop_duplicates().reset_index(drop=True)

    feat_dfs = []
    for name, fn, col, prefix in [
        ('mActivity',    extract_activity, None,      None),
        ('wPedo',        extract_pedo,     None,      None),
        ('wHr',          extract_hr,       None,      None),
        ('mScreenStatus',extract_screen,   None,      None),
        ('mLight',       extract_light,    'm_light', 'mlight'),
        ('wLight',       extract_wlight,   None,      None),
        ('mACStatus',    extract_ac,       None,      None),
        ('mGps',         extract_gps,      None,      None),
        ('mUsageStats',  extract_usage,    None,      None),
        ('mWifi',        extract_wifi,     None,      None),
        ('mBle',         extract_ble,      None,      None),
        ('mAmbience',    extract_ambience, None,      None),
    ]:
        print(f'  {name}...')
        df = load_parquet(name)
        feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))

    sleep_feat_dfs = []
    for name, fn in [
        ('wHr',          extract_sleep_hr),
        ('wPedo',        extract_sleep_pedo),
        ('mActivity',    extract_sleep_activity),
        ('mScreenStatus',extract_sleep_screen),
        ('wLight',       extract_sleep_light),
    ]:
        print(f'  sleep: {name}...')
        df = load_parquet(name)
        sleep_feat_dfs.append(fn(df, sleep_keys))

    sleep_feats = sleep_feat_dfs[0]
    for df in sleep_feat_dfs[1:]:
        sleep_feats = sleep_feats.merge(df, on=['subject_id', 'sleep_date'], how='outer')

    feat_all = feat_dfs[0]
    for df in feat_dfs[1:]:
        feat_all = feat_all.merge(df, on=['subject_id', 'lifelog_date'], how='outer')

    feat_all['dow'] = feat_all['lifelog_date'].dt.dayofweek
    feat_all['month'] = feat_all['lifelog_date'].dt.month
    feat_all['week'] = feat_all['lifelog_date'].dt.isocalendar().week.astype(int)
    feat_all['is_weekend'] = (feat_all['dow'] >= 5).astype(int)
    feat_all['subject_num'] = feat_all['subject_id'].str.extract(r'(\d+)').astype(int)
    feat_all = feat_all.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)

    kr_holidays = holidays.KR(years=2024)
    feat_all['is_holiday'] = feat_all['lifelog_date'].apply(lambda x: 1 if x in kr_holidays else 0)
    feat_all['is_holiday_or_weekend'] = ((feat_all['dow'] >= 5) | (feat_all['is_holiday'] == 1)).astype(int)
    feat_all['next_day'] = feat_all['lifelog_date'] + pd.Timedelta(days=1)
    feat_all['is_next_holiday'] = feat_all['next_day'].apply(lambda x: 1 if x in kr_holidays else 0)
    feat_all['is_next_weekend'] = (feat_all['next_day'].dt.dayofweek >= 5).astype(int)
    feat_all['is_before_freeday'] = ((feat_all['is_next_weekend'] == 1) | (feat_all['is_next_holiday'] == 1)).astype(int)
    feat_all.drop(columns=['next_day'], inplace=True)

    roll_cols = [
        'pedo_total_steps', 'pedo_total_calories', 'pedo_total_distance',
        'screen_on_ratio', 'screen_night_on', 'screen_eve_ratio',
        'act_active_ratio', 'act_still_ratio',
        'mlight_all_mean', 'wlight_all_mean',
        'gps_moving_ratio', 'usage_late_ratio', 'usage_eve_ratio',
        'ac_presleep_charging',
        'hr_mean', 'hr_resting', 'hr_active', 'hr_rmssd',
        'hr_eve_mean', 'hr_night_mean',
    ]
    for col in roll_cols:
        if col not in feat_all.columns:
            continue
        g = feat_all.groupby('subject_id')[col]
        feat_all[f'{col}_lag1'] = g.shift(1)
        feat_all[f'{col}_lag2'] = g.shift(2)
        feat_all[f'{col}_roll3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        feat_all[f'{col}_roll7'] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        feat_all[f'{col}_roll14'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())

    train_date_set = set(zip(train_df['subject_id'], train_df['lifelog_date']))
    is_train = pd.Series(
        [(sid, d) in train_date_set for sid, d in zip(feat_all['subject_id'], feat_all['lifelog_date'])],
        index=feat_all.index,
    )
    exclude_from_norm = {
        'subject_num', 'dow', 'month', 'week', 'is_weekend',
        'is_holiday', 'is_holiday_or_weekend', 'is_next_holiday', 'is_next_weekend', 'is_before_freeday',
    }
    if SUBJ_Z_MODE != 'none':
        norm_cols = [
            c for c in feat_all.select_dtypes(include=[np.number]).columns
            if c not in exclude_from_norm and 'lag' not in c and 'roll' not in c
        ]
        if SUBJ_Z_MODE == 'physio':
            physio_prefixes = ('hr_', 'act_', 'pedo_', 'slp_')
            norm_cols = [c for c in norm_cols if c.startswith(physio_prefixes)]
        for col in norm_cols:
            stats = feat_all.loc[is_train].groupby('subject_id')[col].agg(mu='mean', sig='std')
            stats['sig'] = stats['sig'].replace(0, np.nan)
            mu = feat_all['subject_id'].map(stats['mu'])
            sig = feat_all['subject_id'].map(stats['sig'])
            feat_all[f'{col}_subj_z'] = (feat_all[col] - mu) / sig

    return feat_all, sleep_feats
