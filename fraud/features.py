"""Feature engineering shared by training AND serving (no train/serve skew).

* `add_history_features` – offline: derives velocity / behaviour features from
  the raw transaction log (online, a feature store such as Redis supplies them).
* `add_derived` – pure row-wise features; it is the first step of the sklearn
  pipeline, so the API only needs to send RAW_FEATURES.
"""
import numpy as np
import pandas as pd

from .config import GAP_CAP_MIN, RAW_FEATURES, TARGET


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user rolling features computed ONLY from strictly earlier transactions."""
    df = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    t = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().to_numpy()
    amt = df["amount"].to_numpy(float)
    uid = df["user_id"].to_numpy()
    n = len(df)
    c1, c24 = np.zeros(n), np.zeros(n)
    avg, gap = np.full(n, np.nan), np.full(n, float(GAP_CAP_MIN))

    bounds = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        ts, a = t[s:e], amt[s:e]
        i = np.arange(e - s)
        cum = np.r_[0.0, np.cumsum(a)]
        i1 = np.searchsorted(ts, ts - 3600, "left")
        i24 = np.searchsorted(ts, ts - 86400, "left")
        i30 = np.searchsorted(ts, ts - 30 * 86400, "left")
        c1[s:e], c24[s:e] = i - i1, i - i24
        n30 = i - i30
        avg[s:e] = np.where(n30 > 0, (cum[i] - cum[i30]) / np.maximum(n30, 1), np.nan)
        gap[s:e] = np.minimum(np.r_[GAP_CAP_MIN, np.diff(ts) / 60.0], GAP_CAP_MIN)

    df["txn_count_1h"], df["txn_count_24h"] = c1.astype(int), c24.astype(int)
    # New customers have no history -> fall back to the current amount (ratio = 1).
    df["avg_amount_30d"] = np.where(np.isnan(avg), amt, avg).round(2)
    df["time_since_last_txn_min"] = gap.round(2)
    return df


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_history_features(raw)
    df["hour"] = df["timestamp"].dt.hour
    cols = ["transaction_id", "user_id", "timestamp", *RAW_FEATURES, TARGET]
    return df[cols].sort_values("timestamp").reset_index(drop=True)


def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features; first step of the model pipeline."""
    X = X.copy()
    X["log_amount"] = np.log1p(X["amount"])
    X["is_night"] = ((X["hour"] < 6) | (X["hour"] >= 23)).astype(int)
    X["amount_to_avg_ratio"] = X["amount"] / (X["avg_amount_30d"] + 1.0)
    return X
