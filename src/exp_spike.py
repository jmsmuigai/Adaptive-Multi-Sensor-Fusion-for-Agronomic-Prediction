"""
exp_spike.py -- local spiking experiment (reviewer experiment 1).

Leave-site-out folds (8). For each fold, k samples (k = 0, 5, 10, 20, 50)
from the withheld site group are added to the training set. Spike sets are
NESTED (the k=5 set is contained in the k=10 set, ...) and every k is scored on
the SAME test samples: the withheld samples minus the 50 spike candidates and
minus any sample sharing a plot (top/subsoil pair) with them. Three random
draws. Hyper-parameters: those selected for the fold in the main run.
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

KS = [0, 5, 10, 20, 50]
MODELS = ['PLSR', 'KRR', 'LightGBM', 'Hybrid']
REPS = 3


def job(args):
    t, m, f = args[:3]
    only_k = args[3] if len(args) > 3 else None
    X, wnb, meta, Y = load()
    A = ancillary(meta, np.arange(len(meta)))
    y = Y[t].values.astype(float)
    ok = np.isfinite(y)
    folds = outer_folds('site', meta)
    held = np.intersect1d(folds[f], np.where(ok)[0])
    tr0 = np.setdiff1d(np.where(ok)[0], folds[f])
    params = engine.main_params('site', t, m, f)
    rows = []
    for r in range(1 if m == 'Hybrid' else REPS):
        rng = np.random.default_rng(100 + r)
        cand = rng.permutation(held)[:max(KS)]
        plots = set(meta.PlotKey.values[cand])
        test = np.array([i for i in held if meta.PlotKey.values[i] not in plots])
        if len(test) < 15:
            continue
        for k in (KS if only_k is None else [only_k]):
            tr = np.concatenate([tr0, cand[:k]])
            if m == 'Hybrid':
                cfg = params or __import__('run_main').BASE_H
                P = engine.neural('Hybrid', dict(cfg), X, A, y, tr, test, meta,
                                  inner_design='site', seeds=(11,))
                pred = P[0]
            else:
                pred, _ = engine.classical(m, X, A, y, tr, test, meta, params=params)
            mm = metrics(y[test], pred)
            rows.append(dict(target=t, model=m, fold=f, rep=r, k=k, R2=mm['R2'],
                             RMSE=mm['RMSE'], n_test=len(test),
                             idx=test.tolist(), pred=[float(v) for v in pred]))
    return rows


def jobs():
    return ([(t, m, f) for m in MODELS if m != 'Hybrid' for t in TARGETS for f in range(8)] +
            [(t, 'Hybrid', f, k) for t in TARGETS for f in range(8) for k in KS])


if __name__ == '__main__':
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else MODELS
    jobs = [(t, m, f) for m in only for t in TARGETS for f in range(8)]
    with Pool(2) as p:
        rows = sum(p.map(job, jobs, chunksize=1), [])
    tag = '_'.join(only)
    json.dump(rows, open(os.path.join(OUT, f'exp_spike_{tag}.json'), 'w'))
    df = pd.DataFrame(rows).drop(columns=['idx', 'pred'])
    print(df.groupby(['model', 'target', 'k']).R2.mean().unstack().round(3))
