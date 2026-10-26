"""
cv.py -- Validation designs and evaluation metrics.

Three designs are compared, which is the core experiment of the study:

  random   : plain k-fold over samples. Ignores the fact that samples are
             clustered within survey sites -> optimistic.
  grouped  : GroupKFold on the `site` identifier. No site appears in both
             train and test. Removes within-site memorisation.
  spatial  : contiguous spatial blocks in the (anonymised) coordinate space,
             assigned to folds so that test blocks are geographically
             separated from training blocks.

All preprocessing (imputation, scaling, transforms) is fitted inside the
training fold only and applied to the test fold, via the Preprocessor class.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Fold generators
# ----------------------------------------------------------------------------

def random_folds(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]


def grouped_folds(groups, k=5, seed=0):
    """GroupKFold: greedily assign groups (largest first) to the lightest fold."""
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    load = np.zeros(k, dtype=int)
    assign = {}
    for gi in order:
        f = int(np.argmin(load))
        assign[uniq[gi]] = f
        load[f] += counts[gi]
    fold_of = np.array([assign[g] for g in groups])
    return [np.where(fold_of == f)[0] for f in range(k)]


def spatial_block_folds(coords, k=5, n_blocks=10, seed=0):
    """Partition space into an n_blocks x n_blocks lattice, then assign whole
    blocks to folds at random. Test blocks are therefore spatially disjoint
    from training blocks."""
    rng = np.random.default_rng(seed)
    x, y = coords[:, 0], coords[:, 1]
    bx = np.clip(((x - x.min()) / (np.ptp(x) + 1e-12) * n_blocks).astype(int), 0, n_blocks - 1)
    by = np.clip(((y - y.min()) / (np.ptp(y) + 1e-12) * n_blocks).astype(int), 0, n_blocks - 1)
    bid = bx * n_blocks + by
    uniq = np.unique(bid)
    perm = rng.permutation(len(uniq))
    block_fold = {int(u): int(perm[i] % k) for i, u in enumerate(uniq)}
    fold_of = np.array([block_fold[int(b)] for b in bid])
    return [np.where(fold_of == f)[0] for f in range(k)], bid


DESIGNS = ('random', 'grouped', 'spatial')


def make_folds(design, n, groups=None, coords=None, k=5, seed=0):
    if design == 'random':
        return random_folds(n, k, seed)
    if design == 'grouped':
        return grouped_folds(groups, k, seed)
    if design == 'spatial':
        return spatial_block_folds(coords, k, seed=seed)[0]
    raise ValueError(design)


# ----------------------------------------------------------------------------
# Preprocessing -- fitted on train fold only
# ----------------------------------------------------------------------------

class Preprocessor:
    """Median imputation + z-score standardisation for X; optional log1p for y.

    Fitted exclusively on training-fold rows, which is what the reviewer
    required (no statistic may be estimated using test-fold data).
    """

    def __init__(self, log_target=False):
        self.log_target = log_target

    def fit(self, X, y=None):
        self.med_ = np.nanmedian(X, axis=0)
        Xi = np.where(np.isnan(X), self.med_, X)
        self.mu_ = Xi.mean(0)
        self.sd_ = Xi.std(0)
        self.sd_[self.sd_ < 1e-12] = 1.0
        if y is not None:
            yt = np.log1p(np.maximum(y, 0)) if self.log_target else y
            self.ymu_, self.ysd_ = yt.mean(), yt.std()
            if self.ysd_ < 1e-12:
                self.ysd_ = 1.0
        return self

    def transform_X(self, X):
        Xi = np.where(np.isnan(X), self.med_, X)
        return (Xi - self.mu_) / self.sd_

    def transform_y(self, y):
        yt = np.log1p(np.maximum(y, 0)) if self.log_target else y
        return (yt - self.ymu_) / self.ysd_

    def inverse_y(self, z):
        yt = z * self.ysd_ + self.ymu_
        return np.expm1(yt) if self.log_target else yt


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = y_true.mean() if abs(y_true.mean()) > 1e-12 else np.nan
    return {
        'R2': 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        'RMSE': rmse,
        'MAE': float(np.mean(np.abs(err))),
        'bias': float(np.mean(err)),
        'nRMSE_pct': float(100.0 * rmse / denom) if denom == denom else np.nan,
        'n': int(len(y_true)),
    }


def summarise(fold_metrics):
    """Mean, SD and 95% CI (t-based, small-sample) across folds."""
    out = {}
    keys = [k for k in fold_metrics[0] if k != 'n']
    kf = len(fold_metrics)
    # two-sided 95% t critical values for df = 1..9
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    for k in keys:
        v = np.array([m[k] for m in fold_metrics], dtype=float)
        mu, sd = float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if kf > 1 else 0.0
        half = tcrit.get(kf - 1, 2.262) * sd / np.sqrt(kf) if kf > 1 else 0.0
        out[k + '_mean'] = mu
        out[k + '_sd'] = sd
        out[k + '_ci_lo'] = mu - half
        out[k + '_ci_hi'] = mu + half
    out['n_folds'] = kf
    return out
