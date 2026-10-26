"""
exp_lc.py -- learning curves separating calibration SIZE from DIVERSITY
(reviewer experiment 2 / section 2.2).

Leave-site-out folds (8), fixed test sets. Training sets are drawn from the
training sites of each fold as:
   m sites  in {2, 3, 4, 6, 8, 10, all}  (random choice of sites)
   n samples in {100, 200, 400, 800}     (random subsample of those sites)
Only feasible (m, n) combinations are run. 3 draws per cell.
Models: PLSR and kernel ridge (KRR) with the hyper-parameters selected for
that fold in the main run, and Cubist (fixed: 20 committees, 5 neighbours).
"""
import json
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, ancillary, metrics, outer_folds, OUT
import engine

MS = [2, 3, 4, 6, 8, 10, 99]
NS = [100, 200, 400, 800]
MODELS = ['PLSR', 'KRR', 'Cubist']
REPS = 3


def job(args):
    t, m, f = args
    X, wnb, meta, Y = load()
    A = ancillary(meta, np.arange(len(meta)))
    y = Y[t].values.astype(float)
    ok = np.isfinite(y)
    folds = outer_folds('site', meta)
    te = np.intersect1d(folds[f], np.where(ok)[0])
    tr0 = np.setdiff1d(np.where(ok)[0], folds[f])
    sites = meta.SiteKey.values
    big = [s for s in np.unique(sites[tr0]) if (sites[tr0] == s).sum() >= 20]
    params = engine.main_params('site', t, m, f) if m != 'Cubist' else {'n_committees': 20, 'neighbors': 5}
    rows = []
    for ms in MS:
        mm_ = min(ms, len(big))
        if ms != 99 and ms > len(big):
            continue
        for n in NS:
            for r in range(REPS):
                rng = np.random.default_rng(1000 * ms + 10 * n + r + f)
                ch = big if ms == 99 else list(rng.choice(big, mm_, replace=False))
                pool_ = tr0[np.isin(sites[tr0], ch)]
                if len(pool_) < n:
                    continue
                tr = np.sort(rng.choice(pool_, n, replace=False))
                pred, _ = engine.classical(m, X, A, y, tr, te, meta, params=params)
                q = metrics(y[te], pred)
                rows.append(dict(target=t, model=m, fold=f, m_sites=len(ch), n=n, rep=r,
                                 R2=q['R2'], RMSE=q['RMSE'], idx=te.tolist(),
                                 pred=[float(v) for v in pred]))
    return rows


def jobs():
    return [(t, m, f) for m in MODELS for t in TARGETS for f in range(8)]


if __name__ == '__main__':
    jobs = [(t, m, f) for m in MODELS for t in TARGETS for f in range(8)]
    with Pool(2) as p:
        rows = sum(p.map(job, jobs, chunksize=1), [])
    json.dump(rows, open(os.path.join(OUT, 'exp_lc.json'), 'w'))
    df = pd.DataFrame(rows).drop(columns=['idx', 'pred'])
    print(df.groupby(['model', 'm_sites', 'n']).R2.median().unstack().round(3))
