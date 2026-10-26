"""
agreement.py -- reviewer request: re-run the round-1 baselines with library
implementations (scikit-learn) under EXACTLY the round-1 protocol (round-1
preprocessing without CO2 mask, fold 0 used for choosing among the same three
configurations, folds 1..k-1 evaluated, spectra only) and compare per
(property, fold) R2 with the round-1 NumPy implementations.
"""
import json
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import TARGETS, XScaler, metrics, static_prep, OUT
from multiprocessing import Pool

GR = {
    'PLSR': [dict(n_components=c) for c in (8, 15, 25)],
    'Ridge': [dict(alpha=a) for a in (1.0, 10.0, 100.0)],
    'RandomForest': [dict(n_estimators=40, max_depth=10, min_samples_leaf=3, max_features=0.3),
                     dict(n_estimators=40, max_depth=14, min_samples_leaf=3, max_features=0.5),
                     dict(n_estimators=40, max_depth=10, min_samples_leaf=8, max_features=0.5)],
    'GradientBoosting': [dict(max_iter=200, learning_rate=0.05, max_depth=3),
                         dict(max_iter=300, learning_rate=0.03, max_depth=3),
                         dict(max_iter=150, learning_rate=0.08, max_depth=5)],
}


def mk(name, p):
    if name == 'PLSR':
        return PLSRegression(scale=False, **p)
    if name == 'Ridge':
        return Ridge(**p)
    if name == 'RandomForest':
        return RandomForestRegressor(random_state=0, n_jobs=1, **p)
    return HistGradientBoostingRegressor(random_state=0, early_stopping=False, **p)


class YP:
    def __init__(self, log):
        self.log = log

    def fit(self, y):
        t = np.log1p(np.maximum(y, 0)) if self.log else y
        self.m, self.s = t.mean(), t.std()
        return self

    def fwd(self, y):
        return ((np.log1p(np.maximum(y, 0)) if self.log else y) - self.m) / self.s

    def inv(self, z):
        t = z * self.s + self.m
        return np.expm1(t) if self.log else t


def job(args):
    design, target, name = args
    z = np.load(os.path.join(common.DATA, 'ea_spectra.npz'))
    X, _ = static_prep(z['X'].astype(float), z['wavenumbers'], deriv=1, window=11,
                       binf=4, order='sg_snv', mask_co2=False)
    meta = pd.read_csv(os.path.join(common.DATA, 'ea_meta.csv'))
    Y = pd.read_csv(os.path.join(common.DATA, 'ea_targets.csv'))
    y = Y[target].values.astype(float)
    ok = np.isfinite(y)
    key = {'cluster': meta.ClusterKey.values, 'site': meta.SiteKey.values,
           'country': meta.Country.values}[design]
    folds = common._greedy(key, {'cluster': 5, 'site': 8, 'country': 4}[design])
    log = bool(pd.Series(y[ok]).skew() > 2.0)

    def ev(p, f):
        te = np.array([i for i in folds[f] if ok[i]])
        tr = np.setdiff1d(np.where(ok)[0], folds[f])
        sc = XScaler().fit(X[tr]); yp = YP(log).fit(y[tr])
        m = mk(name, p).fit(sc(X[tr]), yp.fwd(y[tr]))
        pr = yp.inv(np.ravel(m.predict(sc(X[te]))))
        return metrics(y[te], pr)['R2']

    sc0 = [ev(p, 0) for p in GR[name]]
    best = GR[name][int(np.argmax(sc0))]
    return [(design, target, name, f, ev(best, f)) for f in range(1, len(folds))]


def jobs():
    return [(d, t, m) for d in ['cluster', 'site', 'country'] for t in TARGETS for m in GR]


if __name__ == '__main__':
    jobs = [(d, t, m) for d in ['cluster', 'site', 'country'] for t in TARGETS for m in GR]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        rows = sum(p.map(job, jobs), [])
    lib = pd.DataFrame(rows, columns=['design', 'target', 'model', 'fold', 'R2_lib'])
    old = pd.read_csv(os.path.join(common.DATA, 'fold_metrics.csv'))
    m = lib.merge(old[['design', 'target', 'model', 'fold', 'R2']], on=['design', 'target', 'model', 'fold'])
    m.to_csv(os.path.join(OUT, 'agreement.csv'), index=False)
    print(m.groupby('model').apply(lambda g: pd.Series({
        'r': np.corrcoef(g.R2.clip(-3), g.R2_lib.clip(-3))[0, 1],
        'median_abs_diff': np.median(np.abs(g.R2 - g.R2_lib)),
        'n': len(g)})))
