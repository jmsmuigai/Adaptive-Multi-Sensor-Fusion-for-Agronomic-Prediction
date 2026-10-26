"""
exp_kenya.py -- like-for-like test of calibration breadth (reviewer 2.2).

Test targets are ALWAYS the three Kenyan sentinel sites (leave-one-Kenyan-
site-out), so the test population is identical in every condition:

  K2      train on the other two Kenyan sites only
  K2+EA   other two Kenyan sites + all Ethiopian, Tanzanian, Ugandan samples
  EA      Ethiopia + Tanzania + Uganda only (no Kenyan samples)
  K2+EA_n K2+EA randomly subsampled to the size of K2 (5 draws): same n as K2,
          more sites -> separates sample size from environmental diversity
"""
import json
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, ancillary, metrics, OUT
import engine

MODELS = ['PLSR', 'KRR', 'Cubist', 'Hybrid']


def job(args):
    t, m, si = args[:3]
    only_c = args[3] if len(args) > 3 else None
    X, wnb, meta, Y = load()
    A = ancillary(meta, np.arange(len(meta)))
    y = Y[t].values.astype(float)
    ok = np.isfinite(y)
    ken_sites = sorted(meta[meta.Country == 'Kenya'].SiteKey.unique())
    rows = []
    for s in [ken_sites[si]]:
        te = np.where(ok & (meta.SiteKey.values == s))[0]
        K2 = np.where(ok & (meta.Country.values == 'Kenya') & (meta.SiteKey.values != s))[0]
        EA = np.where(ok & (meta.Country.values != 'Kenya'))[0]
        conds = {'K2': [K2], 'K2+EA': [np.concatenate([K2, EA])], 'EA': [EA]}
        rng = np.random.default_rng(1)
        conds['K2+EA_n'] = [np.sort(rng.choice(np.concatenate([K2, EA]), len(K2), replace=False))
                            for _ in range(5)]
        for c, trs in conds.items():
            if only_c is not None and c != only_c:
                continue
            for r, tr in enumerate(trs):
                if m == 'Hybrid':
                    from run_main import BASE_H
                    P = engine.neural('Hybrid', dict(BASE_H), X, A, y, tr, te, meta,
                                      inner_design='site' if c != 'K2' else 'cluster',
                                      seeds=(11,))
                    rs = [metrics(y[te], p)['R2'] for p in P]
                    r2, rm = float(np.mean(rs)), float(np.mean([metrics(y[te], p)['RMSE'] for p in P]))
                    pred = P.mean(0)
                else:
                    pred, _ = engine.classical(m, X, A, y, tr, te, meta,
                                               inner_design='site' if c != 'K2' else 'cluster')
                    mm = metrics(y[te], pred); r2, rm = mm['R2'], mm['RMSE']
                rows.append(dict(target=t, model=m, test_site=s, cond=c, rep=r,
                                 n_train=len(tr), n_sites=int(meta.SiteKey.iloc[tr].nunique()),
                                 R2=r2, RMSE=rm, idx=te.tolist(), pred=[float(v) for v in pred]))
    return rows


def jobs():
    return ([(t, m, si) for m in MODELS if m != 'Hybrid' for t in TARGETS for si in range(3)] +
            [(t, 'Hybrid', si, c) for t in TARGETS for si in range(3) for c in ['K2', 'K2+EA', 'EA', 'K2+EA_n']])


if __name__ == '__main__':
    jobs = [(t, m) for m in MODELS for t in TARGETS]
    with Pool(2) as p:
        rows = sum(p.map(job, jobs, chunksize=1), [])
    json.dump(rows, open(os.path.join(OUT, 'exp_kenya.json'), 'w'))
    df = pd.DataFrame(rows).drop(columns=['idx', 'pred'])
    print(df.groupby(['model', 'cond']).R2.mean().unstack().round(3))
