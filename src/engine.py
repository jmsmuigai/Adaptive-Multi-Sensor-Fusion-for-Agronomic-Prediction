"""
engine.py -- generic fit/predict on arbitrary train/test index sets, used by
the secondary experiments (spiking, learning curves, Kenya like-for-like,
conformal prediction, preprocessing sensitivity). Tuning, when requested, is
done on a group-disjoint inner split exactly as in run_main.py.
"""
import json
import os

import numpy as np

from common import XScaler, YPrep, ancillary, inner_split, inner_kfold, load, OUT

PARTS = os.path.join(OUT, 'parts')


def r2z(y, p):
    return 1 - np.mean((p - y) ** 2) / max(np.var(y), 1e-12)


def main_params(design, target, model, fold):
    p = os.path.join(PARTS, f'{design}__{target}__{model}__f{fold}.json')
    if os.path.exists(p):
        return json.load(open(p))['params']
    return None


def classical(model, X, A, y, tr, te, meta, inner_design='site', params=None,
              return_z=False):
    from classical import MODELS
    XA = np.hstack([X, A])
    mk, grid_, search = MODELS[model]

    def fit(p, a):
        sc = XScaler().fit(XA[a]); yp = YPrep().fit(y[a])
        m = mk(p); m.fit(sc(XA[a]), yp.fwd(y[a]))
        return m, yp, sc

    if params is None:
        if search == 'none' or len(grid_) == 1:
            params = grid_[0]
        else:
            sc_ = []
            for p in grid_:
                if search == 'kfold':
                    s = []
                    for a_, b_ in inner_kfold(inner_design, meta, tr, 3):
                        m, yp, sc = fit(p, tr[a_])
                        s.append(r2z(yp.fwd(y[tr[b_]]), m.predict(sc(XA[tr[b_]]))))
                    sc_.append(np.mean(s))
                else:
                    a, b = inner_split(inner_design, meta, tr)
                    m, yp, sc = fit(p, a)
                    sc_.append(r2z(yp.fwd(y[b]), m.predict(sc(XA[b]))))
            params = grid_[int(np.argmax(sc_))]
    m, yp, sc = fit(params, tr)
    pz = m.predict(sc(XA[te]))
    if return_z:
        return yp.inv(pz), params, (m, yp, sc)
    return yp.inv(pz), params


def neural(kind, cfg, X, A, y, tr, te, meta, inner_design='site', seeds=(11,),
           val_idx=None, return_nets=False, **kw):
    from nets import train_net, predict
    from run_main import TRAIN_KW
    if val_idx is None:
        trn, va = inner_split(inner_design, meta, tr)
    else:
        trn, va = np.setdiff1d(tr, val_idx), val_idx
    sc = XScaler().fit(X[trn]); asc = XScaler().fit(A[trn])
    yp = YPrep().fit(y[trn])
    out, nets = [], []
    for s in seeds:
        net = train_net(kind, cfg, sc(X[trn]), asc(A[trn]), yp.fwd(y[trn]),
                        sc(X[va]), asc(A[va]), yp.fwd(y[va]), seed=s, **TRAIN_KW)
        out.append(yp.inv(predict(net, sc(X[te]), asc(A[te]))))
        nets.append(net)
    if return_nets:
        return np.array(out), (nets, sc, asc, yp, trn, va)
    return np.array(out)
