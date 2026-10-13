"""Stage 3 – train & compare multiple models x imbalance-handling strategies.

Models      : Logistic Regression, Random Forest, XGBoost
Strategies  : baseline (no handling) | class_weight (cost-sensitive) | SMOTE
Selection is done on the VALIDATION split (chronologically after train).
The test split is never touched here.
"""
import json
import time

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud.config import FIG_DIR, RAW_FEATURES, REPORTS_DIR, TARGET
from fraud.data import load_processed, time_split
from fraud.models import BOOSTING_BACKEND, MODEL_LABELS, boosting_pos_weight, build_preprocessor, make_model
from fraud.resampling import smote_resample
from fraud.threshold import evaluate

MODELS = ["logreg", "random_forest", "xgboost"]
STRATEGIES = ["baseline", "class_weight", "smote"]


def main():
    train, val, _ = time_split(load_processed())
    Xtr, ytr, Xva, yva = train[RAW_FEATURES], train[TARGET].to_numpy(), val[RAW_FEATURES], val[TARGET].to_numpy()
    print(f"train {len(train):,} ({ytr.mean():.2%} fraud) | val {len(val):,} ({yva.mean():.2%} fraud) | boosting backend: {BOOSTING_BACKEND}")

    prep = build_preprocessor().fit(Xtr)
    Ztr, Zva = prep.transform(Xtr), prep.transform(Xva)
    pw = boosting_pos_weight(ytr)
    Zsm, ysm, smote_backend = smote_resample(Ztr, ytr, sampling_strategy=0.2)
    print(f"SMOTE ({smote_backend}): {len(ytr):,} -> {len(ysm):,} rows, fraud share {ysm.mean():.1%}")

    rows, curves = [], {}
    for name in MODELS:
        for strat in STRATEGIES:
            model = make_model(name, strat, pos_weight=pw)
            Z, y = (Zsm, ysm) if strat == "smote" else (Ztr, ytr)
            t0 = time.perf_counter(); model.fit(Z, y); fit_s = time.perf_counter() - t0
            p = model.predict_proba(Zva)[:, 1]
            prec, rec, thr = precision_recall_curve(yva, p)
            f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
            best_t = float(thr[f1.argmax()])
            m = evaluate(yva, p, best_t)
            rows.append({"model": MODEL_LABELS[name], "strategy": strat,
                         "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                         "best_f1": m["f1"], "precision@bestF1": m["precision"],
                         "recall@bestF1": m["recall"], "f1@0.5": evaluate(yva, p, 0.5)["f1"],
                         "fit_seconds": fit_s})
            curves[(MODEL_LABELS[name], strat)] = (rec, prec)
            print(f"{MODEL_LABELS[name]:<20} {strat:<13} PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  F1* {m['f1']:.4f}  fit {fit_s:5.1f}s")

    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    res.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    (REPORTS_DIR / "model_comparison.md").write_text(res.to_markdown(index=False, floatfmt=".4f"))
    (REPORTS_DIR / "model_comparison_meta.json").write_text(json.dumps(
        {"boosting_backend": BOOSTING_BACKEND, "smote_backend": smote_backend,
         "train_fraud_rate": float(ytr.mean()), "val_fraud_rate": float(yva.mean())}, indent=2))

    # ---- figures -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for a, strat in zip(ax, STRATEGIES):
        for (label, s), (rec, prec) in curves.items():
            if s == strat:
                r = res[(res.model == label) & (res.strategy == s)].iloc[0]
                a.plot(rec, prec, label=f"{label} (AP={r.pr_auc:.3f})")
        a.axhline(yva.mean(), ls=":", c="grey"); a.set_title(f"Strategy: {strat}")
        a.set_xlabel("recall"); a.legend(loc="lower left")
    ax[0].set_ylabel("precision"); fig.suptitle("Precision-Recall curves on validation set")
    fig.tight_layout(); fig.savefig(FIG_DIR / "model_comparison_pr_curves.png", dpi=130); plt.close(fig)

    pivot = res.pivot(index="model", columns="strategy", values="pr_auc")[STRATEGIES]
    ax = pivot.plot.bar(figsize=(8, 4.5), rot=0, color=["#8c8c8c", "#4C72B0", "#55A868"])
    ax.set_ylabel("PR-AUC (average precision)"); ax.set_title("PR-AUC by model and imbalance strategy (validation)")
    ax.set_ylim(max(0, pivot.min().min() - 0.1), min(1.0, pivot.max().max() + 0.05))
    plt.tight_layout(); plt.savefig(FIG_DIR / "model_comparison_bar.png", dpi=130); plt.close()
    print("\n", res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
