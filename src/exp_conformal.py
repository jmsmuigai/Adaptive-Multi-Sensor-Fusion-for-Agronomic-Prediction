"""
exp_conformal.py -- split-conformal prediction intervals with a GROUP-AWARE
calibration set (reviewer experiment 7).

Within each outer training fold, a calibration set is held out and not used
for fitting:
   group : whole sentinel sites (about 20% of training samples)
   random: a random 20% of training samples (ignores spatial grouping)
Absolute residuals are computed on the model's fitting scale (log1p for skewed
targets), q = ceil((n+1)(1-alpha))/n empirical quantile, interval
[f(x)-q, f(x)+q] back-transformed. Coverage and median width on the outer test
fold are reported for alpha = 0.10 (nominal 90%).
"""
import json
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, ancillary, outer_folds, inner_split, OUT, NFOLDS, YPrep
import engine

ALPHA = 0.10
MODELS = ['KRR', 'LightGBM', 'Hybrid']
DESIGNS = ['site', 'aez', 'country']


def qhat(res, alpha=ALPHA):
    n = len(res)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(res)[min(k, n) - 1]


def job(args):
    d, t, m, f = args
    X, wnb, meta, Y = load()
    A = ancillary(meta, np.arange(len(meta)))
    y = Y[t].values.astype(float)
    ok = np.isfinite(y)
    folds = outer_folds(d, meta)
    te = np.intersect1d(folds[f], np.where(ok)[0])
    tr = np.setdiff1d(np.where(ok)[0], folds[f])
    params = engine.main_params(d, t, m, f)
    rows = []
    for cal_kind in ['group', 'random']:
        if cal_kind == 'group':
            fit_idx, cal = inner_split('country' if d != 'cluster' else 'cluster', meta, tr,
                                       frac=0.2, seed=99)
        else:
            rng = np.random.default_rng(99)
            p = rng.permutation(tr); nc = int(0.2 * len(tr))
            cal, fit_idx = np.sort(p[:nc]), np.sort(p[nc:])
        both = np.concatenate([cal, te])
        if m == 'Hybrid':
            cfg = params or __import__('run_main').BASE_H
            P = engine.neural('Hybrid', dict(cfg), X, A, y, fit_idx, both, meta,
                              inner_design=d if d != 'random' else 'site', seeds=(11,))[0]
        else:
            P, _ = engine.classical(m, X, A, y, fit_idx, both, meta, params=params)
        yp = YPrep().fit(y[fit_idx])
        tf = (lambda v: np.log1p(np.maximum(v, 0))) if yp.log else (lambda v: v)
        inv = (lambda v: np.expm1(v)) if yp.log else (lambda v: v)
        pc, pt = P[:len(cal)], P[len(cal):]
        q = qhat(np.abs(tf(y[cal]) - tf(pc)))
        lo, hi = inv(tf(pt) - q), inv(tf(pt) + q)
        if yp.log:
            lo = np.maximum(lo, 0)
        cover = (y[te] >= lo) & (y[te] <= hi)
        rows.append(dict(design=d, target=t, model=m, fold=f, cal=cal_kind,
                         n_cal=len(cal), coverage=float(cover.mean()),
                         width_median=float(np.median(hi - lo)),
                         n_test=len(te), cover_vec=cover.astype(int).tolist(),
                         idx=te.tolist()))
    return rows


def jobs():
    return [(d, t, m, f) for m in MODELS for d in DESIGNS for t in TARGETS for f in range(NFOLDS[d])]


if __name__ == '__main__':
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else MODELS
    jobs = [(d, t, m, f) for m in only for d in DESIGNS for t in TARGETS for f in range(NFOLDS[d])]
    with Pool(2) as p:
        rows = sum(p.map(job, jobs, chunksize=1), [])
    tag = '_'.join(only)
    json.dump(rows, open(os.path.join(OUT, f'exp_conformal_{tag}.json'), 'w'))
    df = pd.DataFrame(rows)
    df['n_cov'] = df.coverage * df.n_test
    g = df.groupby(['design', 'model', 'cal']).apply(lambda x: x.n_cov.sum() / x.n_test.sum())
    print(g.unstack().round(3))
