"""
run_main.py -- nested, group-aware benchmark for the revision.

Task = (design, target, model, fold[, variant]). Each task writes one JSON in
out/parts with the test indices, predictions (original units; one vector per
seed for neural models), the configuration selected inside the training fold,
and diagnostics (validation loss, ||W_o||_1, gate, attention).

    python3 run_main.py --stage main --workers 2
    python3 run_main.py --stage ablation --workers 2
"""
import argparse
import json
import os
import sys
import time
import traceback

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load, TARGETS, SEED, outer_folds, inner_split,   # noqa: E402
                    inner_kfold, XScaler, YPrep, ancillary, OUT, NFOLDS)

PARTS = os.path.join(OUT, 'parts')
os.makedirs(PARTS, exist_ok=True)

CLASSICAL = ['PLSR', 'Ridge', 'SVR', 'KRR', 'GPR', 'Cubist', 'MBL', 'RF', 'LightGBM']
NEURAL = ['MLP', 'CNN', 'Hybrid']
SEEDS = [11, 22, 33, 44, 55]

BASE_H = dict(lr=3e-3, bn=False, c1=16, c2=32, k1=11, s1=2, k2=7, att_dim=32,
              D=8, R=16, d_rbf=8, dropout=0.0, wd=1e-4)
SPACE = {
    'Hybrid': [dict(BASE_H),
               dict(BASE_H, c1=32, c2=64, R=32),
               dict(BASE_H, k1=15, s1=3, dropout=0.1, wd=1e-3)],
    'CNN': [dict(lr=3e-3, c1=16, c2=32, k1=11, s1=2, h=64, dropout=0.2),
            dict(lr=1e-3, c1=32, c2=64, k1=11, s1=2, h=64, dropout=0.2),
            dict(lr=3e-3, c1=16, c2=32, k1=15, s1=3, h=128, dropout=0.3, wd=1e-3)],
    'MLP': [dict(lr=1e-3, h1=128, h2=64, dropout=0.1),
            dict(lr=1e-3, h1=256, h2=128, dropout=0.3),
            dict(lr=3e-3, h1=64, h2=32, dropout=0.0, wd=1e-3)],
}
SPACE['MOCNN'] = SPACE['CNN']

# Ablations: exactly one component of the full model replaced
ABLATIONS = {
    'head_linear': dict(head='linear'),
    'head_mlp': dict(head='mlp'),
    'head_gauss': dict(head='gauss'),
    'pool_mean': dict(pool='mean'),
    'pool_reweight': dict(pool='reweight'),
    'no_pos': dict(pos=False),
    'fuse_concat': dict(fuse='concat'),
    'spectra_only': dict(fuse='none'),
    'plus_coords': dict(coords=True),
}
ABL_SEEDS = [11, 22, 33]

TRAIN_KW = dict(max_epochs=80, patience=10, min_epochs=15, batch=64)


def path(d, t, m, f, v=''):
    return os.path.join(PARTS, f'{d}__{t}__{m}{("__" + v) if v else ""}__f{f}.json')


def r2z(y, p):
    return 1 - np.mean((p - y) ** 2) / max(np.var(y), 1e-12)


def split_data(design, target, fold):
    X, wnb, meta, Y = load()
    folds = outer_folds(design, meta)
    if target == 'ALL':
        ok = np.ones(len(meta), bool)
    else:
        ok = np.isfinite(Y[target].values)
    te = np.intersect1d(folds[fold], np.where(ok)[0])
    tr = np.setdiff1d(np.where(ok)[0], folds[fold])
    return X, meta, Y, tr, te


# ------------------------------------------------------------ classical
def run_classical(design, target, model, fold, coords=False):
    from classical import MODELS
    X, meta, Y, tr, te = split_data(design, target, fold)
    y = Y[target].values.astype(float)
    A = ancillary(meta, np.arange(len(meta)), coords=coords)
    XA = np.hstack([X, A])
    mk, grid_, search = MODELS[model]

    def fit_pred(params, a, b, lgb_val=None):
        sc = XScaler().fit(XA[a]); yp = YPrep().fit(y[a])
        m = mk(params)
        if model == 'LightGBM' and lgb_val is not None:
            m.fit(sc(XA[a]), yp.fwd(y[a]), sc(XA[lgb_val]), yp.fwd(y[lgb_val]))
        else:
            m.fit(sc(XA[a]), yp.fwd(y[a]))
        return m, yp, sc

    scores = []
    best_iter = {}
    if search == 'none' or len(grid_) == 1:
        best = grid_[0]
    else:
        for ci, p in enumerate(grid_):
            if search == 'kfold':
                s = []
                for a_, b_ in inner_kfold(design, meta, tr, 3):
                    a, b = tr[a_], tr[b_]
                    m, yp, sc = fit_pred(p, a, b)
                    s.append(r2z(yp.fwd(y[b]), m.predict(sc(XA[b]))))
                scores.append(float(np.mean(s)))
            else:
                a, b = inner_split(design, meta, tr)
                m, yp, sc = fit_pred(p, a, b, lgb_val=b if model == 'LightGBM' else None)
                scores.append(float(r2z(yp.fwd(y[b]), m.predict(sc(XA[b])))))
                if model == 'LightGBM':
                    best_iter[ci] = int(m.best_iter or 200)
        bi = int(np.argmax(scores))
        best = dict(grid_[bi])
        if model == 'LightGBM':
            best['n_estimators'] = max(50, int(1.1 * best_iter[bi]))
    m, yp, sc = fit_pred(best, tr, None)
    pz = m.predict(sc(XA[te]))
    pred = yp.inv(pz)
    return {'params': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in best.items()},
            'inner_scores': scores, 'log_target': yp.log,
            'test_idx': te.tolist(), 'pred': [float(v) for v in pred]}


# --------------------------------------------------------------- neural
def run_neural(design, target, model, fold, variant='', seeds=SEEDS):
    from nets import train_net, predict
    X, meta, Y, tr, te = split_data(design, target, fold)
    override = dict(ABLATIONS.get(variant, {}))
    coords = override.pop('coords', False)
    A = ancillary(meta, np.arange(len(meta)), coords=coords)
    trn, va = inner_split(design, meta, tr)
    sc = XScaler().fit(X[trn])
    asc = XScaler().fit(A[trn])
    kind = 'CNN' if model == 'MOCNN' else model
    if target == 'ALL':
        yps = [YPrep().fit(Y[t].values[trn][np.isfinite(Y[t].values[trn])]) for t in TARGETS]
        Yz = np.column_stack([yps[j].fwd(Y[t].values) for j, t in enumerate(TARGETS)])
        ytr, yva = Yz[trn], Yz[va]
    else:
        y = Y[target].values.astype(float)
        yp = YPrep().fit(y[trn])
        ytr, yva = yp.fwd(y[trn]), yp.fwd(y[va])
    Xtr, Xva, Xte = sc(X[trn]), sc(X[va]), sc(X[te])
    Atr, Ava, Ate = asc(A[trn]), asc(A[va]), asc(A[te])

    # configuration: tuned on the inner split for the main run; for an
    # ablation, the configuration selected for the full model is reused
    tune = []
    if variant:
        base = json.load(open(path(design, target, 'Hybrid', fold)))
        cfg = dict(base['params']); cfg.update(override)
    import pickle
    ck = path(design, target, model, fold, variant) + '.ckpt'
    C = pickle.load(open(ck, 'rb')) if os.path.exists(ck) else {'tune': [], 'seeds': {}}

    def save_ck():
        pickle.dump(C, open(ck + '.tmp', 'wb')); os.replace(ck + '.tmp', ck)
    if not variant:
        for ci, c in enumerate(SPACE[model]):
            if ci < len(C['tune']):
                continue
            net = train_net(kind, c, Xtr, Atr, ytr, Xva, Ava, yva, seed=seeds[0], **TRAIN_KW)
            C['tune'].append(float(net.val_loss)); save_ck()
        tune = C['tune']
        cfg = dict(SPACE[model][int(np.argmin(tune))])

    preds, info = [], []
    for s in seeds:
        if s in C['seeds']:
            rec, pv = C['seeds'][s]
            info.append(rec); preds.append(pv)
            continue
        net = train_net(kind, cfg, Xtr, Atr, ytr, Xva, Ava, yva, seed=s, **TRAIN_KW)
        pz = predict(net, Xte, Ate)
        rec = {'seed': s, 'val_loss': float(net.val_loss), 'best_epoch': int(net.best_epoch)}
        if kind == 'Hybrid':
            rec['wo_l1'] = net.wo_l1()
            rec['bo'] = float(net.out.bias.item())
            rec['rf_bins'] = int(net.rf); rec['jump_bins'] = int(net.jump)
            import torch
            net._store = True
            with torch.no_grad():
                net(torch.as_tensor(Xte, dtype=torch.float32),
                    torch.as_tensor(Ate, dtype=torch.float32))
            if net.cache['G'] is not None:
                rec['gate_spec'] = float(net.cache['G'][:, 0].mean())
            if net.cache['alpha'] is not None and not variant:
                rec['attention'] = net.cache['alpha'].mean(0).numpy().round(6).tolist()
            # range of the head output over the test fold (standardised scale)
            rec['pz_min'], rec['pz_max'] = float(pz.min()), float(pz.max())
            if not variant and design in ('site', 'cluster') and s == seeds[0]:
                torch.save(net.state_dict(), path(design, target, model, fold).replace('.json', '.pt'))
        info.append(rec)
        if target == 'ALL':
            preds.append({t: [float(v) for v in yps[j].inv(pz[:, j])] for j, t in enumerate(TARGETS)})
        else:
            preds.append([float(v) for v in yp.inv(pz)])
        C['seeds'][s] = (rec, preds[-1]); save_ck()
    if os.path.exists(ck):
        os.remove(ck)
    out = {'params': cfg, 'tune_val': tune, 'test_idx': te.tolist(), 'seeds': info}
    if target == 'ALL':
        out['pred_by_target'] = {t: [p[t] for p in preds] for t in TARGETS}
    else:
        out['pred'] = preds
        out['log_target'] = yp.log
    return out


def run_task(task):
    d, t, m, f, v = task
    p = path(d, t, m, f, v)
    if os.path.exists(p):
        return 'skip'
    t0 = time.time()
    try:
        if m in NEURAL or m == 'MOCNN':
            sd = SEEDS if (not v or (v == 'head_linear' and d != 'site') or v == 'head_linear') else ABL_SEEDS
            res = run_neural(d, t, m, f, v, seeds=sd)
        else:
            res = run_classical(d, t, m, f, coords=(v == 'plus_coords'))
    except Exception:
        err = traceback.format_exc()
        print('FAILED', task, err[-800:], flush=True)
        return 'fail'
    res.update({'design': d, 'target': t, 'model': m, 'fold': f, 'variant': v,
                'runtime_s': round(time.time() - t0, 1)})
    json.dump(res, open(p, 'w'))
    return f'{d}/{t}/{m}/{v}/f{f} {res["runtime_s"]}s'


def tasks_for(stage):
    out = []
    if stage == 'main':
        for d in ['site', 'country', 'aez', 'cluster', 'random']:
            for m in ['PLSR', 'Ridge', 'SVR', 'KRR', 'Cubist', 'MBL', 'GPR', 'LightGBM',
                      'RF', 'MLP', 'Hybrid', 'CNN']:
                if d == 'random' and m == 'CNN':
                    continue
                for t in TARGETS:
                    for f in range(NFOLDS[d]):
                        out.append((d, t, m, f, ''))
            if d != 'random':
                for f in range(NFOLDS[d]):
                    out.append((d, 'ALL', 'MOCNN', f, ''))
    elif stage == 'ablation':
        for v in ABLATIONS:
            for t in TARGETS:
                for f in range(NFOLDS['site']):
                    out.append(('site', t, 'Hybrid', f, v))
        for d in ['country', 'aez']:
            for t in TARGETS:
                for f in range(NFOLDS[d]):
                    out.append((d, t, 'Hybrid', f, 'head_linear'))
        for t in TARGETS:
            for f in range(NFOLDS['site']):
                out.append(('site', t, 'LightGBM', f, 'plus_coords'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='main')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--only', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--seeds', default='')
    a = ap.parse_args()
    todo = [t for t in tasks_for(a.stage) if not os.path.exists(path(*t))]
    if a.only:
        keep = a.only.split(',')
        todo = [t for t in todo if t[2] in keep or t[0] in keep]
    if a.reverse:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(todo)} tasks', flush=True)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=50) as p:
        for i, r in enumerate(p.imap_unordered(run_task, todo, chunksize=1)):
            if i % 25 == 0:
                print(f'[{i}/{len(todo)} {time.time() - t0:.0f}s] {r}', flush=True)
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
