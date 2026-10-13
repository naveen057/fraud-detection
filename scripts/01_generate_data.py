"""Stage 1 – data collection & preprocessing.

raw transactions -> validation/cleaning -> history features -> processed CSV.
"""
import json

import _common  # noqa: F401
import pandas as pd

from fraud.config import DATA_DIR, PROCESSED_PATH, RAW_PATH, REPORTS_DIR, TARGET
from fraud.data import generate_transactions
from fraud.features import build_training_frame


def main():
    DATA_DIR.mkdir(exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True)

    print("Collecting (simulating) raw transactions ...")
    raw = generate_transactions()
    raw.to_csv(RAW_PATH, index=False)

    # ---- data-quality checks & cleaning ------------------------------------
    report = {"raw_rows": int(len(raw))}
    report["missing_values"] = int(raw.isna().sum().sum())
    n_dup = int(raw.duplicated(subset=[c for c in raw.columns if c != "transaction_id"]).sum())
    report["duplicate_rows_removed"] = n_dup
    raw = raw.drop_duplicates(subset=[c for c in raw.columns if c != "transaction_id"])
    report["non_positive_amounts_removed"] = int((raw["amount"] <= 0).sum())
    raw = raw[raw["amount"] > 0]

    # ---- feature engineering -----------------------------------------------
    df = build_training_frame(raw)
    df.to_csv(PROCESSED_PATH, index=False)

    report.update({
        "processed_rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "fraud_count": int(df[TARGET].sum()),
        "fraud_rate_pct": round(100 * df[TARGET].mean(), 3),
        "imbalance_ratio": f"1 : {int((1 - df[TARGET].mean()) / df[TARGET].mean())}",
        "columns": list(df.columns),
    })
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
