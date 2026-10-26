"""
exp_prep.py -- sensitivity of transfer skill to spectral preprocessing
(reviewer: preprocessing order, spectral range / CO2 region, alternatives).

Leave-site-out, PLSR and KRR, nested tuning as in the main run.
"""
import json
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import TARGETS, ancillary, metrics, outer_folds, OUT, static_prep
import engine

VARIANTS = {
    'SG1+SNV, CO2 masked (adopted)': dict(deriv=1, window=11, binf=4, order='sg_snv', mask_co2=True),
    'SG1+SNV, unmasked (round 1)': dict(deriv=1, window=11, binf=4, order='sg_snv', mask_co2=False),
    'SNV then SG1, masked': dict(deriv=1, window=11, binf=4, order='snv_sg', mask_co2=True),
    'SNV only, masked': dict(deriv=None, window=11, binf=4, order='snv', mask_co2=True),
    'SG2+SNV, masked': dict(deriv=2, window=15, binf=4, order='sg_snv', mask_co2=True),
    'SG1+SNV, masked, bin x8': dict(deriv=1, window=11, binf=8, order='sg_snv', mask_co2=True),
}
_X = {}


def job(args):
    v, t, m = args
    _, _, meta, Y = common.load()
    if v not in _X:
        _X[v] = static_prep(common._C['Xraw'], common._C['wn'], **VARIANTS[v])[0]
    X = _X[v]
    A = ancillary(meta, np.arange(len(meta)))
    y = Y[t].values.astype(float)
    ok = np.isfinite(y)
    folds = outer_folds('site', meta)
    idx, pr = [], []
    for f in range(8):
        te = np.intersect1d(folds[f], np.where(ok)[0])
        tr = np.setdiff1d(np.where(ok)[0], folds[f])
        p, _ = engine.classical(m, X, A, y, tr, te, meta, inner_design='site')
        idx.append(te); pr.append(p)
    idx, pr = np.concatenate(idx), np.concatenate(pr)
    q = metrics(y[idx], pr)
    return dict(variant=v, target=t, model=m, R2=q['R2'], RMSE=q['RMSE'], RPIQ=q['RPIQ'])


def jobs():
    return [(v, t, m) for v in VARIANTS for m in ['PLSR', 'KRR'] for t in TARGETS]


if __name__ == '__main__':
    jobs = [(v, t, m) for v in VARIANTS for m in ['PLSR', 'KRR'] for t in TARGETS]
    with Pool(2) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'exp_prep.csv'), index=False)
    print(df.pivot_table(index='variant', columns='model', values='R2', aggfunc='mean').round(3))
