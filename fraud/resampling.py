"""SMOTE wrapper: imbalanced-learn when installed, tiny NumPy fallback otherwise."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import SEED


def smote_resample(X, y, sampling_strategy=0.2, k=5, seed=SEED):
    """Oversample the minority class to `sampling_strategy` x majority count.

    Applied to the TRAINING split only – validation/test stay untouched.
    Returns (X_res, y_res, backend_name).
    """
    X, y = np.asarray(X), np.asarray(y)
    try:
        from imblearn.over_sampling import SMOTE
        Xr, yr = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k,
                       random_state=seed).fit_resample(X, y)
        return Xr, yr, "imbalanced-learn"
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    n_new = int(sampling_strategy * (y == 0).sum()) - len(minority)
    if n_new <= 0:
        return X, y, "numpy-fallback"
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(minority))).fit(minority)
    neigh = nn.kneighbors(minority, return_distance=False)[:, 1:]
    base = rng.integers(0, len(minority), n_new)
    pick = neigh[base, rng.integers(0, neigh.shape[1], n_new)]
    lam = rng.random((n_new, 1))
    synth = minority[base] + lam * (minority[pick] - minority[base])
    return (np.vstack([X, synth]),
            np.concatenate([y, np.ones(n_new, dtype=y.dtype)]),
            "numpy-fallback")
