"""Stage 4 – hyper-parameter tuning, threshold optimisation, final evaluation.

1. RandomizedSearchCV (TimeSeriesSplit, scoring = PR-AUC) for the boosting model and Random Forest.
2. Re-fit the tuned models with class-weighting AND with SMOTE; pick the winner on VALIDATION PR-AUC.
3. Choose the decision threshold on VALIDATION by minimising business cost.
4. Report ONCE on the held-out chronological TEST split.
5. Benchmark single-transaction inference latency; save model bundle + metrics.json.

Use --fast for a quick smoke run (fewer search iterations).
"""
import argparse
import json
import time

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from fraud.config import FIG_DIR, MODEL_PATH, RAW_FEATURES, REPORTS_DIR, REVIEW_COST_PER_ALERT, SEED, TARGET
from fraud.data import load_processed, time_split
from fraud.models import (BOOSTING_BACKEND, MODEL_LABELS, SEARCH_SPACES, boosting_pos_weight,
                          build_preprocessor, make_model)
from fraud.predictor import FraudPredictor, save_bundle
from fraud.resampling import smote_resample
from fraud.threshold import cost_curve, evaluate, expected_cost, optimize_threshold

sns.set_theme(style="whitegrid")


def tune(name, Xtr, ytr, n_iter):
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", make_model(name, "class_weight", pos_weight=boosting_pos_weight(ytr)))])
    space = {f"clf__{k}": v for k, v in SEARCH_SPACES[name].items()}
    rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision",
                            cv=TimeSeriesSplit(n_splits=3), random_state=SEED, n_jobs=1,
                            refit=False, verbose=1)
    t0 = time.perf_counter(); rs.fit(Xtr, ytr)
    best = {k.replace("clf__", ""): v for k, v in rs.best_params_.items()}
    print(f"[{MODEL_LABELS[name]}] best CV PR-AUC {rs.best_score_:.4f} in {time.perf_counter()-t0:.0f}s -> {best}")
    return best, float(rs.best_score_)


def main(fast=False):
    train, val, test = time_split(load_processed())
    Xtr, ytr = train[RAW_FEATURES], train[TARGET].to_numpy()
    Xva, yva = val[RAW_FEATURES], val[TARGET].to_numpy()
    Xte, yte = test[RAW_FEATURES], test[TARGET].to_numpy()
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} (fraud {yte.mean():.2%}) | backend {BOOSTING_BACKEND}")

    # ---- 1. tuning ---------------------------------------------------------
    n_iter = {"xgboost": 3 if fast else 8, "random_forest": 2 if fast else 5}
    tuned = {n: tune(n, Xtr, ytr, n_iter[n]) for n in ["xgboost", "random_forest"]}

    # ---- 2. candidates: tuned model x imbalance strategy -------------------
    prep = build_preprocessor().fit(Xtr)
    Ztr, Zva = prep.transform(Xtr), prep.transform(Xva)
    Zsm, ysm, smote_backend = smote_resample(Ztr, ytr, sampling_strategy=0.2)
    pw = boosting_pos_weight(ytr)
    cands = []
    for name, (params, cv_score) in tuned.items():
        for strat in ["class_weight", "smote"]:
            clf = make_model(name, strat, pos_weight=pw, **params)
            Z, y = (Zsm, ysm) if strat == "smote" else (Ztr, ytr)
            clf.fit(Z, y)
            p = clf.predict_proba(Zva)[:, 1]
            m = evaluate(yva, p, 0.5)
            cands.append({"name": name, "strategy": strat, "params": params, "clf": clf,
                          "val_pr_auc": m["pr_auc"], "val_roc_auc": m["roc_auc"], "cv_pr_auc": cv_score, "val_p": p})
            print(f"  candidate {MODEL_LABELS[name]:<14} {strat:<12} val PR-AUC {m['pr_auc']:.4f}")
    best = max(cands, key=lambda c: c["val_pr_auc"])
    print(f"==> winner: {MODEL_LABELS[best['name']]} + {best['strategy']}")
    pipeline = Pipeline([("prep", prep), ("clf", best["clf"])])

    # ---- 3. threshold optimisation (validation only) -----------------------
    p_val = best["val_p"]
    opt = optimize_threshold(yva, p_val, val["amount"].to_numpy())
    thr = opt["cost_threshold"]
    print(f"thresholds -> cost-optimal {thr:.4f} | F1-optimal {opt['f1_threshold']:.4f} | default 0.5")

    # ---- 4. final test evaluation ------------------------------------------
    p_te = pipeline.predict_proba(Xte)[:, 1]
    amt_te = test["amount"].to_numpy()
    final = {}
    for label, t in [("default_0.5", 0.5), ("f1_optimal", opt["f1_threshold"]), ("cost_optimal", thr)]:
        m = evaluate(yte, p_te, t)
        m["business_cost"] = expected_cost(yte, p_te, amt_te, t)
        final[label] = m
        print(f"  TEST @{label:<12} thr {t:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  cost ${m['business_cost']:,.0f}")
    no_model = float(amt_te[yte == 1].sum())

    # ---- 5. latency benchmark ----------------------------------------------
    predictor = FraudPredictor({"pipeline": pipeline, "threshold": thr})
    recs = Xte.head(300).to_dict("records")
    for r in recs[:20]: predictor.predict([r])                      # warm-up
    lat = []
    for r in recs:
        t0 = time.perf_counter(); predictor.predict([r]); lat.append((time.perf_counter() - t0) * 1000)
    batch = Xte.head(1000).to_dict("records"); t0 = time.perf_counter(); predictor.predict(batch)
    batch_s = time.perf_counter() - t0
    latency = {"single_p50_ms": float(np.percentile(lat, 50)), "single_p95_ms": float(np.percentile(lat, 95)),
               "single_p99_ms": float(np.percentile(lat, 99)), "single_mean_ms": float(np.mean(lat)),
               "batch_1000_total_ms": batch_s * 1000, "batch_throughput_tps": 1000 / batch_s}
    print("latency:", {k: round(v, 2) for k, v in latency.items()})

    # ---- figures -----------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    prec, rec, _ = precision_recall_curve(yte, p_te)
    ax[0].plot(rec, prec, lw=2, label=f"AP = {final['cost_optimal']['pr_auc']:.3f}")
    for label, c in [("default_0.5", "grey"), ("f1_optimal", "#55A868"), ("cost_optimal", "#C44E52")]:
        ax[0].scatter(final[label]["recall"], final[label]["precision"], s=80, c=c, zorder=3,
                      label=f"{label} (t={final[label]['threshold']:.2f})")
    ax[0].axhline(yte.mean(), ls=":", c="grey"); ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
    ax[0].set_title("Test Precision-Recall curve"); ax[0].legend()
    fpr, tpr, _ = roc_curve(yte, p_te)
    ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {final['cost_optimal']['roc_auc']:.3f}"); ax[1].plot([0, 1], [0, 1], "k:")
    ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate"); ax[1].set_title("Test ROC curve"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "final_pr_roc.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, label in zip(ax, ["default_0.5", "f1_optimal", "cost_optimal"]):
        m = final[label]
        sns.heatmap([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], annot=True, fmt=",d", cbar=False, cmap="Blues", ax=a,
                    xticklabels=["Pred legit", "Pred fraud"], yticklabels=["Actual legit", "Actual fraud"])
        a.set_title(f"{label}  (t={m['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(FIG_DIR / "final_confusion_matrices.png", dpi=130); plt.close(fig)

    grid, costs = cost_curve(yva, p_val, val["amount"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, costs, lw=2); ax.axvline(thr, c="#C44E52", ls="--", label=f"cost-optimal t={thr:.3f}")
    ax.axvline(0.5, c="grey", ls=":", label="default t=0.5"); ax.set_xlabel("decision threshold")
    ax.set_ylabel(f"validation cost ($)  [${REVIEW_COST_PER_ALERT:.0f}/alert + missed fraud $]")
    ax.set_title("Threshold optimisation (validation set)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "final_threshold_cost.png", dpi=130); plt.close(fig)

    pos = np.random.default_rng(SEED).choice(len(Xva), size=min(8000, len(Xva)), replace=False)
    pi = permutation_importance(pipeline, Xva.iloc[pos], yva[pos], scoring="average_precision",
                                n_repeats=3, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=RAW_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_xlabel("drop in PR-AUC when feature is shuffled"); ax.set_title("Permutation feature importance (validation)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "final_feature_importance.png", dpi=130); plt.close(fig)

    # ---- save ---------------------------------------------------------------
    metrics = {
        "boosting_backend": BOOSTING_BACKEND, "smote_backend": smote_backend,
        "winner": {"model": MODEL_LABELS[best["name"]], "strategy": best["strategy"], "params": best["params"]},
        "candidates": [{"model": MODEL_LABELS[c["name"]], "strategy": c["strategy"],
                        "cv_pr_auc": c["cv_pr_auc"], "val_pr_auc": c["val_pr_auc"]} for c in cands],
        "threshold": {"selected": thr, "f1_optimal": opt["f1_threshold"], "review_cost_per_alert": REVIEW_COST_PER_ALERT},
        "test": final, "test_n": int(len(test)), "test_fraud_n": int(yte.sum()),
        "test_cost_if_no_model": no_model, "latency": latency,
        "permutation_importance": imp.sort_values(ascending=False).round(5).to_dict(),
    }
    (REPORTS_DIR / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_bundle(MODEL_PATH, pipeline, thr, MODEL_LABELS[best["name"]], best["strategy"],
                metrics={"test_pr_auc": final["cost_optimal"]["pr_auc"], "test_recall": final["cost_optimal"]["recall"],
                         "test_precision": final["cost_optimal"]["precision"]},
                extra={"boosting_backend": BOOSTING_BACKEND})
    print(f"\nSaved model -> {MODEL_PATH} ({MODEL_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true")
    main(ap.parse_args().fast)
