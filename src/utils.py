import random
import numpy as np


def seed_everything(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else np.nan


def safe_sum(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sum(x)) if len(x) else 0.0


def safe_max(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if len(x) else np.nan


def safe_entropy(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def clip_probs(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.5, posinf=1.0 - eps, neginf=eps)
    return np.clip(x, eps, 1.0 - eps)


def agg_stats(vals, prefix):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return {f'{prefix}_{k}': np.nan for k in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']}
    return {
        f'{prefix}_mean': np.nanmean(vals),
        f'{prefix}_std': np.nanstd(vals),
        f'{prefix}_min': np.nanmin(vals),
        f'{prefix}_max': np.nanmax(vals),
        f'{prefix}_median': np.nanmedian(vals),
        f'{prefix}_q25': np.nanpercentile(vals, 25),
        f'{prefix}_q75': np.nanpercentile(vals, 75),
    }


def parse_hr(v, lo=40, hi=200):
    try:
        if isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v, dtype=float).ravel()
        else:
            arr = np.array([float(v)])
        arr = arr[(arr >= lo) & (arr <= hi)]
    except Exception:
        arr = np.array([])
    return arr
