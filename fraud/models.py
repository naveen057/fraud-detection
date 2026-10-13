"""Model factory, preprocessing pipeline and hyper-parameter search spaces.

XGBoost is the primary boosting model.  If `xgboost` is not importable the
code transparently falls back to sklearn's HistGradientBoostingClassifier
(same algorithm family) so the pipeline still runs; the backend used is
recorded in every report.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORICAL, NUMERIC, SEED
from .features import add_derived

try:  # pragma: no cover - depends on environment
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

BOOSTING_BACKEND = "xgboost" if HAS_XGB else "sklearn-hist-gb-fallback"
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost" if HAS_XGB else "XGBoost*",
}


def build_preprocessor() -> Pipeline:
    """raw API features -> derived features -> scaled numeric + one-hot categorical."""
    encode = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("derive", FunctionTransformer(add_derived, validate=False)),
                     ("encode", encode)])


def make_model(name: str, strategy: str = "baseline", pos_weight: float = 1.0, **params):
    """strategy: 'baseline' (nothing), 'class_weight' (cost-sensitive) or 'smote'
    (data already resampled -> no weights)."""
    weighted = strategy == "class_weight"
    if name == "logreg":
        p = dict(max_iter=1000, C=1.0, class_weight="balanced" if weighted else None,
                 random_state=SEED)
        return LogisticRegression(**{**p, **params})
    if name == "random_forest":
        p = dict(n_estimators=150, max_depth=12, min_samples_leaf=5, n_jobs=-1,
                 class_weight="balanced_subsample" if weighted else None, random_state=SEED)
        return RandomForestClassifier(**{**p, **params})
    if name == "xgboost":
        w = pos_weight if weighted else 1.0
        if HAS_XGB:
            p = dict(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                     colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                     scale_pos_weight=w, n_jobs=-1, random_state=SEED)
            return XGBClassifier(**{**p, **params})
        p = dict(max_iter=300, learning_rate=0.1, max_depth=6,
                 class_weight={0: 1.0, 1: w} if weighted else None, random_state=SEED)
        return HistGradientBoostingClassifier(**{**p, **params})
    raise ValueError(name)


SEARCH_SPACES = {
    "random_forest": {
        "n_estimators": [100, 150, 200],
        "max_depth": [8, 12, 16],
        "min_samples_leaf": [2, 5, 10],
        "max_features": ["sqrt", 0.5],
    },
    "xgboost": (
        {"n_estimators": [200, 300, 500], "max_depth": [3, 4, 5, 6, 8],
         "learning_rate": [0.03, 0.05, 0.1, 0.2], "subsample": [0.7, 0.85, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
         "gamma": [0, 0.1, 0.5]}
        if HAS_XGB else
        {"max_iter": [150, 250, 400], "max_depth": [3, 4, 6, 8],
         "learning_rate": [0.03, 0.05, 0.1, 0.2], "min_samples_leaf": [20, 50, 100],
         "l2_regularization": [0.0, 0.1, 1.0], "max_leaf_nodes": [15, 31, 63]}
    ),
}


def boosting_pos_weight(y) -> float:
    """sqrt(neg/pos): a moderate re-weighting (the full ratio over-fires)."""
    y = np.asarray(y)
    return float(np.sqrt((y == 0).sum() / max((y == 1).sum(), 1)))
