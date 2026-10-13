import numpy as np

from fraud.config import RAW_FEATURES, TARGET
from fraud.features import add_derived
from fraud.predictor import FraudPredictor
from fraud.resampling import smote_resample
from fraud.threshold import evaluate, expected_cost, optimize_threshold


def test_data_is_imbalanced_and_clean(small_frame):
    rate = small_frame[TARGET].mean()
    assert 0.005 < rate < 0.05
    assert small_frame[RAW_FEATURES].isna().sum().sum() == 0
    assert (small_frame["amount"] > 0).all()


def test_history_features_use_only_the_past(small_frame):
    first = small_frame.sort_values("timestamp").groupby("user_id").head(1)
    assert (first["txn_count_1h"] == 0).all() and (first["txn_count_24h"] == 0).all()


def test_derived_features():
    import pandas as pd
    X = pd.DataFrame({"amount": [100.0], "hour": [3], "avg_amount_30d": [49.0]})
    out = add_derived(X)
    assert out["is_night"].iloc[0] == 1 and np.isclose(out["amount_to_avg_ratio"].iloc[0], 2.0)


def test_smote_only_adds_minority():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 4)); y = np.zeros(1000, int); y[:30] = 1
    Xr, yr, _ = smote_resample(X, y, sampling_strategy=0.3)
    assert len(Xr) == len(yr) > 1000 and (yr == 0).sum() == 970 and yr.sum() > 30


def test_threshold_optimisation_beats_nothing_and_default():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    p = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.12, 5000), 0, 1)
    amt = rng.uniform(10, 300, 5000)
    opt = optimize_threshold(y, p, amt)
    assert opt["cost_at_cost_threshold"] <= expected_cost(y, p, amt, 0.5)
    assert opt["cost_at_cost_threshold"] < opt["cost_if_no_model"]
    assert evaluate(y, p, opt["cost_threshold"])["recall"] > 0.5


def test_predictor_output_contract(model_path, sample_txn):
    pred = FraudPredictor.load(model_path)
    rec = {k: sample_txn[k] for k in RAW_FEATURES}
    out = pred.predict([rec, rec])
    assert len(out) == 2 and 0 <= out[0]["fraud_probability"] <= 1
    assert out[0]["risk_level"] in {"low", "medium", "high"}


def test_risky_beats_benign(model_path, sample_txn):
    pred = FraudPredictor.load(model_path)
    risky = {k: sample_txn[k] for k in RAW_FEATURES}
    benign = {**risky, "amount": 12.0, "hour": 14, "merchant_category": "grocery", "channel": "pos",
              "is_foreign": 0, "device_new": 0, "distance_from_home_km": 2.0, "avg_amount_30d": 15.0}
    p = pred.predict_proba([risky, benign])
    assert p[0] > p[1]
