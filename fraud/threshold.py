"""Metrics and decision-threshold optimisation.

With ~1.5 % fraud, the default 0.5 cut-off is almost never optimal.  We pick the
threshold on the VALIDATION set by minimising an explicit business cost:

    cost = REVIEW_COST x (#alerts)  +  sum(amount of fraud we missed)
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import REVIEW_COST_PER_ALERT


def evaluate(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(threshold),
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "alert_rate": float(pred.mean())}


def expected_cost(y, p, amount, threshold, review_cost=REVIEW_COST_PER_ALERT) -> float:
    y, p, amount = np.asarray(y), np.asarray(p), np.asarray(amount)
    flagged = p >= threshold
    missed = amount[(~flagged) & (y == 1)].sum()
    return float(review_cost * flagged.sum() + missed)


def _grid(p):
    return np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 99),
        np.quantile(p, np.linspace(0.90, 0.9999, 300)),
    ]))


def cost_curve(y, p, amount, review_cost=REVIEW_COST_PER_ALERT):
    grid = _grid(np.asarray(p))
    return grid, np.array([expected_cost(y, p, amount, t, review_cost) for t in grid])


def optimize_threshold(y, p, amount, review_cost=REVIEW_COST_PER_ALERT) -> dict:
    """Return the cost-optimal and the F1-optimal thresholds (fit on validation)."""
    grid, costs = cost_curve(y, p, amount, review_cost)
    t_cost = float(grid[int(np.argmin(costs))])
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    t_f1 = float(thr[int(np.argmax(f1))])
    return {"cost_threshold": t_cost, "f1_threshold": t_f1,
            "cost_at_cost_threshold": float(costs.min()),
            "cost_if_no_model": float(np.asarray(amount)[np.asarray(y) == 1].sum())}
