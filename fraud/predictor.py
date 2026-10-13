"""Framework-independent inference wrapper (easy to unit-test, used by the API)."""
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import RAW_FEATURES


def save_bundle(path, pipeline, threshold, model_name, strategy, metrics=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline, "threshold": float(threshold),
        "model_name": model_name, "strategy": strategy,
        "raw_features": RAW_FEATURES, "metrics": metrics or {},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }, path, compress=3)


class FraudPredictor:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.info = {k: v for k, v in bundle.items() if k != "pipeline"}

    @classmethod
    def load(cls, path) -> "FraudPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. Run `make train` (or scripts/04_tune_and_evaluate.py).")
        return cls(joblib.load(path))

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        X = pd.DataFrame.from_records(records)[RAW_FEATURES]
        return self.pipeline.predict_proba(X)[:, 1]

    def risk_level(self, p: float) -> str:
        return "high" if p >= self.threshold else "medium" if p >= self.threshold / 2 else "low"

    def predict(self, records: list[dict]) -> list[dict]:
        probs = self.predict_proba(records)
        return [{"fraud_probability": round(float(p), 6),
                 "is_fraud": bool(p >= self.threshold),
                 "risk_level": self.risk_level(float(p)),
                 "threshold": self.threshold} for p in probs]
