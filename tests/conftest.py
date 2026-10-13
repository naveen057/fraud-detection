import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from sklearn.pipeline import Pipeline

from fraud.config import RAW_FEATURES, TARGET
from fraud.data import generate_transactions
from fraud.features import build_training_frame
from fraud.models import build_preprocessor, make_model
from fraud.predictor import save_bundle


@pytest.fixture(scope="session")
def small_frame():
    return build_training_frame(generate_transactions(n_users=300, days=30, seed=7))


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, small_frame):
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", make_model("random_forest", "class_weight", n_estimators=30, max_depth=8))])
    pipe.fit(small_frame[RAW_FEATURES], small_frame[TARGET])
    path = tmp_path_factory.mktemp("model") / "m.joblib"
    save_bundle(path, pipe, 0.3, "Random Forest", "class_weight")
    return path


@pytest.fixture
def sample_txn():
    return {"transaction_id": "t1", "amount": 899.99, "hour": 2, "merchant_category": "electronics",
            "channel": "online", "is_foreign": 1, "device_new": 1, "distance_from_home_km": 340.0,
            "account_age_days": 420, "txn_count_1h": 1, "txn_count_24h": 2,
            "avg_amount_30d": 48.5, "time_since_last_txn_min": 35.0}
