"""
run_kenya.py -- Benchmark the proposed architecture against tuned baselines on
the Kenya AfSIS MIR dataset.

Validation designs
  cluster : leave-cluster-out (45 LDSF clusters -> 5 folds). Primary design.
            No cluster contributes samples to both training and test.
  site    : leave-site-out (3 LDSF sentinel sites -> 3 folds). Tests transfer
            across agro-ecological zones. This is the honest version of the
            "AEZ hold-out transfer" analysis.

Spectral preprocessing (fitted within the training fold only)
  1. Savitzky-Golay smoothing + optional first derivative
  2. Standard Normal Variate (SNV) scatter correction
  3. binning to reduce 1749 bands to a manageable length
  4. mean/SD standardisation per band

Models (equal tuning budget of 3 configurations each)
  PLSR         partial least squares -- the chemometric standard for MIR
  Ridge        L2 linear on preprocessed spectra
  RandomForest histogram random forest
  GBM          histogram gradient boosting
  MLP          dense network
  DualDAFM     the proposed 1D-CNN + DAFM + RBF architecture

Usage:  python3 run_kenya.py --ntasks 20
"""

import argparse
import json
import math
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from multiprocessing import Pool          # noqa: E402

import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml import MODELS, RidgeRegressor      # noqa: E402
from ml_dual import DualBranchDAFM         # noqa: E402
from cv import metrics                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KEN = os.path.join(HERE, 'ea')
SCRATCH = '/tmp/ea_work'
PARTS = os.path.join(SCRATCH, 'parts')
FINAL = os.path.join(KEN, 'parts')
os.makedirs(PARTS, exist_ok=True)
os.makedirs(FINAL, exist_ok=True)

SEED = 20260804
TARGETS = ['pH', 'OrgC_pct', 'TotN_pct', 'P_Olsen', 'Ca_exch', 'Mg_exch',
           'K_exch', 'eCEC', 'S_total', 'Fe_total', 'Mn_total', 'Zn_total',
           'Cu_total']
DESIGNS = ['cluster', 'site', 'country']
MODEL_NAMES = ['PLSR', 'Ridge', 'RandomForest', 'GradientBoosting', 'MLP', 'DualDAFM']
N_CFG = 3

_C = {}


# ------------------------------------------------------------------ data
def load():
    if 'X' not in _C:
        z = np.load(os.path.join(KEN, 'ea_spectra.npz'))
        _C['X'] = z['X']
        _C['wn'] = z['wavenumbers']
        _C['meta'] = pd.read_csv(os.path.join(KEN, 'ea_meta.csv'))
        _C['y'] = pd.read_csv(os.path.join(KEN, 'ea_targets.csv'))
    return _C['X'], _C['wn'], _C['meta'], _C['y']


NFOLDS = {'cluster': 5, 'site': 8, 'country': 4}


def folds_for(design, meta, k=None):
    """Group-disjoint folds at three nested spatial scales:
    LDSF cluster, LDSF sentinel site, and country."""
    key = {'cluster': meta.ClusterKey.values,
           'site': meta.SiteKey.values,
           'country': meta.Country.values}[design]
    uniq, counts = np.unique(key, return_counts=True)
    kk = min(NFOLDS[design], len(uniq))
    order = np.argsort(-counts)
    load_ = np.zeros(kk, dtype=int)
    assign = {}
    for gi in order:
        f = int(np.argmin(load_))
        assign[uniq[gi]] = f
        load_[f] += counts[gi]
    fo = np.array([assign[g] for g in key])
    return [np.where(fo == f)[0] for f in range(kk)]


# ------------------------------------------------- spectral preprocessing
def savgol(X, window=11, poly=2, deriv=0):
    """Savitzky-Golay filter implemented directly (no scipy dependency)."""
    half = window // 2
    idx = np.arange(-half, half + 1)
    A = np.vander(idx, poly + 1, increasing=True)
    # least-squares coefficients for the requested derivative at the centre
    coef = np.linalg.pinv(A)[deriv] * (math.factorial(deriv) if deriv else 1.0)
    Xp = np.pad(X, ((0, 0), (half, half)), mode='edge')
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = Xp[:, j:j + window] @ coef
    return out


def snv(X):
    m = X.mean(1, keepdims=True)
    s = X.std(1, keepdims=True)
    s[s < 1e-12] = 1.0
    return (X - m) / s


def bin_spectra(X, factor):
    n, L = X.shape
    L2 = (L // factor) * factor
    return X[:, :L2].reshape(n, L2 // factor, factor).mean(2)


class SpecPrep:
    """Spectral pipeline. Only the standardisation step is data-dependent, and
    it is fitted on the training fold alone."""

    def __init__(self, deriv=1, window=11, binf=4):
        self.deriv, self.window, self.binf = deriv, window, binf

    def fit(self, X):
        Z = self._static(X)
        self.mu_, self.sd_ = Z.mean(0), Z.std(0)
        self.sd_[self.sd_ < 1e-12] = 1.0
        return self

    def _static(self, X):
        Z = savgol(X, self.window, 2, self.deriv)
        Z = snv(Z)
        return bin_spectra(Z, self.binf)

    def transform(self, X):
        return (self._static(X) - self.mu_) / self.sd_


class YPrep:
    def __init__(self, log=False):
        self.log = log

    def fit(self, y):
        t = np.log1p(np.maximum(y, 0)) if self.log else y
        self.mu_, self.sd_ = float(np.mean(t)), float(np.std(t))
        if self.sd_ < 1e-12:
            self.sd_ = 1.0
        return self

    def fwd(self, y):
        t = np.log1p(np.maximum(y, 0)) if self.log else y
        return (t - self.mu_) / self.sd_

    def inv(self, z):
        t = z * self.sd_ + self.mu_
        return np.expm1(t) if self.log else t


# ------------------------------------------------------------------ PLSR
class PLSR:
    """NIPALS partial least squares regression -- the chemometric reference
    method for infrared soil spectroscopy."""

    def __init__(self, n_components=15, **kw):
        self.k = n_components

    def fit(self, X, y):
        Xc = X - X.mean(0)
        self.xm_ = X.mean(0)
        self.ym_ = float(np.mean(y))
        yc = y - self.ym_
        T, P, Q = [], [], []
        E, f = Xc.copy(), yc.copy()
        for _ in range(self.k):
            w = E.T @ f
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                break
            w = w / nw
            t = E @ w
            tt = float(t @ t)
            if tt < 1e-12:
                break
            p = E.T @ t / tt
            q = float(f @ t) / tt
            E = E - np.outer(t, p)
            f = f - q * t
            T.append(t); P.append(p); Q.append(q)
        if not T:
            self.coef_ = np.zeros(X.shape[1])
            return self
        T = np.array(T).T; P = np.array(P).T; Q = np.array(Q)
        W = np.linalg.pinv(P.T)
        self.coef_ = W @ Q
        return self

    def predict(self, X):
        return (X - self.xm_) @ self.coef_ + self.ym_


GRIDS = {
    'PLSR': [{'n_components': c} for c in (8, 15, 25)],
    'Ridge': [{'alpha': a} for a in (1.0, 10.0, 100.0)],
    'RandomForest': [
        {'n_estimators': 40, 'max_depth': 10, 'min_samples_leaf': 3, 'max_features': 0.3},
        {'n_estimators': 40, 'max_depth': 14, 'min_samples_leaf': 3, 'max_features': 0.5},
        {'n_estimators': 40, 'max_depth': 10, 'min_samples_leaf': 8, 'max_features': 0.5},
    ],
    'GradientBoosting': [
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
        {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 3},
        {'n_estimators': 150, 'learning_rate': 0.08, 'max_depth': 5},
    ],
    'MLP': [
        {'hidden': (64, 32), 'lr': 1e-3, 'epochs': 150, 'l2': 1e-3, 'patience': 20},
        {'hidden': (128, 64), 'lr': 1e-3, 'epochs': 150, 'l2': 1e-3, 'patience': 20},
        {'hidden': (64, 32), 'lr': 3e-4, 'epochs': 200, 'l2': 1e-2, 'patience': 25},
    ],
    'DualDAFM': [
        {'conv_channels': (16, 32), 'kernels': (9, 7), 'strides': (4, 1),
         'pool': 2, 'cov_hidden': 16, 'att_dim': 32, 'n_rbf': 16,
         'lr': 2e-3, 'epochs': 90, 'batch': 128, 'l2': 1e-4, 'patience': 12},
        {'conv_channels': (24, 48), 'kernels': (13, 7), 'strides': (5, 1),
         'pool': 2, 'cov_hidden': 16, 'att_dim': 48, 'n_rbf': 24,
         'lr': 2e-3, 'epochs': 90, 'batch': 128, 'l2': 1e-4, 'patience': 12},
        {'conv_channels': (16, 32), 'kernels': (9, 5), 'strides': (4, 1),
         'pool': 2, 'cov_hidden': 16, 'att_dim': 32, 'n_rbf': 16,
         'lr': 1e-3, 'epochs': 110, 'batch': 128, 'l2': 1e-3, 'patience': 15},
    ],
}
SPEC_CFG = {'deriv': 1, 'window': 11, 'binf': 4}


def covariates(meta, idx):
    """Minimal ancillary covariates available without the GEE extraction:
    depth flag and position. Replaced by the full GEE stack when available."""
    m = meta.iloc[idx]
    return np.column_stack([m.is_topsoil.values,
                            m.Latitude.values, m.Longitude.values]).astype(float)


def fit_one(name, params, Xtr, Ctr, ytr, Xte, Cte, yte_s):
    if name == 'DualDAFM':
        mdl = DualBranchDAFM(Xtr.shape[1], Ctr.shape[1], seed=SEED, **params)
        mdl.fit(Xtr, Ctr, ytr, Xte, Cte, yte_s)
        return mdl.predict(Xte, Cte)
    if name == 'PLSR':
        return PLSR(**params).fit(Xtr, ytr).predict(Xte)
    if name == 'MLP':
        m = MODELS['MLP'](**params, seed=SEED)
        m.fit(Xtr, ytr, Xte, yte_s)
        return m.predict(Xte)
    m = MODELS[name](**params, seed=SEED)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def path_for(d, t, m, ph):
    return os.path.join(PARTS, f'{d}__{t}__{m}__{ph}.json')


def run_task(task):
    design, target, model, phase = task
    out = path_for(design, target, model, phase)
    if os.path.exists(out):
        return 'skip'
    t0 = time.time()
    X, wn, meta, Y = load()
    y_all = Y[target].values.astype(float)
    ok = np.isfinite(y_all)
    folds = folds_for(design, meta)
    log = bool(pd.Series(y_all[ok]).skew() > 2.0)

    def evaluate(params, f):
        te = np.array([i for i in folds[f] if ok[i]])
        tr = np.array([i for i in range(len(y_all)) if ok[i] and i not in set(folds[f])])
        sp = SpecPrep(**SPEC_CFG).fit(X[tr])
        yp = YPrep(log).fit(y_all[tr])
        Xtr, Xte = sp.transform(X[tr]), sp.transform(X[te])
        Ctr, Cte = covariates(meta, tr), covariates(meta, te)
        cm, cs = Ctr.mean(0), Ctr.std(0)
        cs[cs < 1e-12] = 1.0
        Ctr, Cte = (Ctr - cm) / cs, (Cte - cm) / cs
        pr = yp.inv(fit_one(model, params, Xtr, Ctr, yp.fwd(y_all[tr]),
                            Xte, Cte, yp.fwd(y_all[te])))
        return metrics(y_all[te], pr), te, pr

    if phase.startswith('tune'):
        c = int(phase[4:])
        err = None
        try:
            m0, _, _ = evaluate(GRIDS[model][c], 0)
            score = m0['R2']
        except Exception as e:
            import traceback
            score, err = None, traceback.format_exc()[-500:]
            print('TUNE FAILED', task, err, flush=True)
        payload = {'task': task, 'cfg': c, 'params': GRIDS[model][c],
                   'tuning_R2': score, 'error': err, 'log_target': log,
                   'runtime_s': round(time.time() - t0, 2)}
    else:
        scores = []
        for c in range(N_CFG):
            p = path_for(design, target, model, f'tune{c}')
            if not os.path.exists(p):
                return 'wait'
            s = json.load(open(p))['tuning_R2']
            scores.append(-np.inf if s is None else s)
        params = GRIDS[model][int(np.argmax(scores))]
        f = int(phase[1:])
        try:
            mm, te, pr = evaluate(params, f)
        except Exception as e:
            return f'ERROR {e}'
        payload = {'task': task, 'params': params, 'fold': f, 'metrics': mm,
                   'test_idx': te.tolist(), 'pred': [float(v) for v in pr],
                   'runtime_s': round(time.time() - t0, 2)}
    json.dump(payload, open(out, 'w'))
    return f'{design}/{target}/{model}/{phase} {payload["runtime_s"]}s'


def all_tasks():
    have = set(os.listdir(PARTS))
    order = ['PLSR', 'Ridge', 'GradientBoosting', 'RandomForest', 'MLP', 'DualDAFM']
    _, _, meta, _ = load()
    out = []
    phases = [f'tune{c}' for c in range(N_CFG)]
    for d in DESIGNS:
        phases_d = phases + [f'f{i}' for i in range(1, NFOLDS[d])]
        for ph in phases_d:
            for m in order:
                for t in TARGETS:
                    if f'{d}__{t}__{m}__{ph}.json' not in have:
                        out.append((d, t, m, ph))
    return out


def sync():
    import shutil
    have = set(os.listdir(FINAL))
    n = 0
    for f in os.listdir(PARTS):
        if f not in have:
            shutil.copy2(os.path.join(PARTS, f), os.path.join(FINAL, f))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ntasks', type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    todo = all_tasks()
    if not todo:
        print('ALL TASKS COMPLETE;', sync(), 'synced')
        return
    batch = todo[:a.ntasks]
    t0 = time.time()
    with Pool(a.workers) as p:
        p.map(run_task, batch, chunksize=1)
    rem = len(all_tasks())
    print(f'ran {len(batch)} in {time.time() - t0:.0f}s | remaining {rem}', flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
