"""Central configuration: paths, feature lists, business-cost assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RAW_PATH = DATA_DIR / "transactions_raw.csv"
PROCESSED_PATH = DATA_DIR / "transactions_processed.csv"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"

SEED = 42
TARGET = "is_fraud"

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "retail", "travel",
    "entertainment", "electronics", "jewelry", "gift_cards", "crypto",
]
CHANNELS = ["online", "pos", "atm"]

# Features the API receives (what a feature store / upstream service provides).
RAW_FEATURES = [
    "amount", "hour", "merchant_category", "channel", "is_foreign", "device_new",
    "distance_from_home_km", "account_age_days",
    "txn_count_1h", "txn_count_24h", "avg_amount_30d", "time_since_last_txn_min",
]

# Features after `features.add_derived` (what the model sees).
NUMERIC = [
    "amount", "log_amount", "hour", "is_night", "is_foreign", "device_new",
    "distance_from_home_km", "account_age_days",
    "txn_count_1h", "txn_count_24h", "avg_amount_30d",
    "amount_to_avg_ratio", "time_since_last_txn_min",
]
CATEGORICAL = ["merchant_category", "channel"]

# Business-cost assumptions used for decision-threshold optimisation.
# Missing a fraud costs the transaction amount; every alert costs analyst time.
REVIEW_COST_PER_ALERT = 5.0

# Cap for "no previous transaction" gaps (minutes) = 7 days.
GAP_CAP_MIN = 7 * 24 * 60
