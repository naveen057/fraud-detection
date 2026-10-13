"""Synthetic-but-realistic transaction generator + loading / splitting helpers.

Why synthetic?  Real card data is private.  The generator reproduces the
properties that make fraud detection hard: extreme class imbalance, bursts of
card-testing, account takeovers and "subtle" fraud that looks like normal
behaviour.  To use real data (e.g. the Kaggle ULB dataset) replace
`generate_transactions` with a loader that returns the same columns.
"""
import numpy as np
import pandas as pd

from .config import CHANNELS, MERCHANT_CATEGORIES, PROCESSED_PATH, SEED

START = pd.Timestamp("2025-01-01")

_HOUR_W = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 10, 11, 10, 9, 9, 9, 10, 10, 9, 7, 5, 3, 2], float)
_HOUR_LEGIT = _HOUR_W / _HOUR_W.sum()
_HOUR_NIGHT_W = np.array([8, 9, 9, 8, 6, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 6, 7], float)
_HOUR_FRAUD = _HOUR_NIGHT_W / _HOUR_NIGHT_W.sum()

_CAT_LEGIT = np.array([.28, .20, .12, .15, .06, .08, .07, .015, .015, .01])
_CAT_FRAUD = np.array([.04, .04, .03, .12, .07, .05, .30, .12, .15, .08])
_CAT_SUBTLE = 0.5 * _CAT_LEGIT + 0.5 * _CAT_FRAUD


def _frame(uid, ts, amount, cat, chan, foreign, new_dev, dist, users, fraud):
    return pd.DataFrame({
        "user_id": uid,
        "ts_sec": ts,
        "amount": amount,
        "merchant_category": cat,
        "channel": chan,
        "is_foreign": foreign.astype(int),
        "device_new": new_dev.astype(int),
        "distance_from_home_km": dist,
        "account_age_days": users["account_age_days"].to_numpy()[uid],
        "is_fraud": fraud,
    })


def generate_transactions(n_users=2500, days=60, fraud_rate=0.016, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({
        "avg_amount": rng.lognormal(np.log(45), 0.6, n_users),
        "account_age_days": rng.integers(30, 3650, n_users),
    })
    lam = rng.gamma(4, 10.0, n_users) * (days / 60)  # ≈ 40 txns / user / 60 days

    # ---- legitimate traffic -------------------------------------------------
    counts = rng.poisson(lam)
    uid = np.repeat(np.arange(n_users), counts)
    n = len(uid)
    ts = (rng.integers(0, days, n) * 86400
          + rng.choice(24, n, p=_HOUR_LEGIT) * 3600 + rng.uniform(0, 3600, n))
    legit = _frame(
        uid, ts,
        users["avg_amount"].to_numpy()[uid] * rng.lognormal(-0.2, 0.7, n),
        rng.choice(MERCHANT_CATEGORIES, n, p=_CAT_LEGIT),
        rng.choice(CHANNELS, n, p=[.35, .55, .10]),
        rng.random(n) < 0.03, rng.random(n) < 0.05,
        np.where(rng.random(n) < 0.9, rng.exponential(8, n), rng.exponential(80, n)),
        users, np.zeros(n, int))

    # ---- fraud --------------------------------------------------------------
    n_fraud = int(fraud_rate * n / (1 - fraud_rate))
    n_burst, n_take = int(.25 * n_fraud), int(.40 * n_fraud)
    n_subtle = n_fraud - n_burst - n_take
    blocks = []

    # 1) card-testing bursts: many tiny online payments within minutes
    for _ in range(max(n_burst // 6, 1)):
        k = int(rng.integers(4, 9))
        u = int(rng.integers(0, n_users))
        t0 = rng.integers(0, days) * 86400 + rng.choice(24, p=_HOUR_FRAUD) * 3600 + rng.uniform(0, 3600)
        t = t0 + np.cumsum(rng.uniform(20, 150, k))
        blocks.append(_frame(
            np.full(k, u), t, rng.uniform(1, 15, k),
            rng.choice(["retail", "gift_cards", "electronics", "entertainment"], k),
            rng.choice(CHANNELS, k, p=[.9, .08, .02]),
            rng.random(k) < .25, rng.random(k) < .8,
            rng.lognormal(np.log(60), 1.0, k), users, np.ones(k, int)))

    # 2) account takeover: big, foreign / new device, night-time
    m = n_take
    u = rng.integers(0, n_users, m)
    blocks.append(_frame(
        u, rng.integers(0, days, m) * 86400 + rng.choice(24, m, p=_HOUR_FRAUD) * 3600 + rng.uniform(0, 3600, m),
        users["avg_amount"].to_numpy()[u] * rng.lognormal(1.3, 0.7, m),
        rng.choice(MERCHANT_CATEGORIES, m, p=_CAT_FRAUD),
        rng.choice(CHANNELS, m, p=[.7, .2, .1]),
        rng.random(m) < .35, rng.random(m) < .55,
        rng.lognormal(np.log(120), 1.0, m), users, np.ones(m, int)))

    # 3) subtle fraud: looks almost like normal behaviour (the hard cases)
    m = n_subtle
    u = rng.integers(0, n_users, m)
    blocks.append(_frame(
        u, rng.integers(0, days, m) * 86400 + rng.choice(24, m, p=_HOUR_LEGIT) * 3600 + rng.uniform(0, 3600, m),
        users["avg_amount"].to_numpy()[u] * rng.lognormal(0.0, 0.7, m),
        rng.choice(MERCHANT_CATEGORIES, m, p=_CAT_SUBTLE),
        rng.choice(CHANNELS, m, p=[.5, .4, .1]),
        rng.random(m) < .08, rng.random(m) < .20,
        rng.exponential(15, m), users, np.ones(m, int)))

    df = pd.concat([legit, *blocks], ignore_index=True)
    df["amount"] = df["amount"].clip(lower=0.5).round(2)
    df["distance_from_home_km"] = df["distance_from_home_km"].round(1)
    df["timestamp"] = START + pd.to_timedelta(df.pop("ts_sec"), unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(len(df)))
    return df


def load_processed(path=PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def time_split(df: pd.DataFrame, train=0.70, val=0.15):
    """Chronological split – never train on the future (mirrors production)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    i, j = int(len(df) * train), int(len(df) * (train + val))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()
