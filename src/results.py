"""
results.py -- build every number, table and figure used in the revised
manuscript from the stored per-fit records. Writes out/numbers.json,
out/tables/*.csv and out/fig/*.png.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load, TARGETS, LABEL, OUT, NFOLDS, outer_folds, metrics, AEZ_NAME,
                    SITE_AEZ, YPrep)
import analyze as A

B = int(os.environ.get("NBOOT", 2000))
TAB = os.path.join(OUT, 'tables'); os.makedirs(TAB, exist_ok=True)
EXP = [os.path.join(OUT, 'vm', 'exp'), os.path.join(OUT, 'exp')]
DESIGNS = ['random', 'cluster', 'site', 'aez', 'country']
DNAME = {'random': 'Random', 'cluster': 'Leave-cluster-out', 'site': 'Leave-site-out',
         'aez': 'Leave-AEZ-out', 'country': 'Leave-country-out'}
MODELS = A.MODELS_ORDER
N = {}


def exp_rows(name):
    rows, seen = [], set()
    for base in EXP:
        for f in glob.glob(os.path.join(base, name, '*.json')):
            k = os.path.basename(f)
            if k in seen:
                continue
            seen.add(k)
            r = json.load(open(f))
            rows += r if isinstance(r, list) else [r]
    return rows


def clip_out(Y, t, idx, p):
    """limit predictions to the range of the target outside the test set"""
    y = Y[t].values
    m = np.ones(len(y), bool); m[idx] = False; m &= np.isfinite(y)
    return np.clip(p, y[m].min(), y[m].max())


def r2v(y, P):
    """R2 for each row of P (S, n)."""
    return 1 - ((P - y) ** 2).sum(1) / ((y - y.mean()) ** 2).sum()


def main_tables(parts, X, meta, Y):
    rows, preds = [], {}
    for d in DESIGNS:
        for m in MODELS:
            U = A.assemble(parts, d, m, clip=False)
            for t, (idx, P, fo) in A.assemble(parts, d, m).items():
                o = np.argsort(idx); idx, P, fo = idx[o], P[:, o], fo[o]
                tm, ts = A.train_stats(d, t)
                r = A.pooled_metrics(Y[t].values[idx], P, fo, tm, ts)
                iu, Pu, _ = U[t]; ou = np.argsort(iu)
                yu = Y[t].values[iu[ou]]
                r['R2_unclipped'] = float(r2v(yu, Pu[:, ou]).mean())
                r['R2log_unclipped'] = float(r2v(np.log1p(np.maximum(yu, 0)), np.log1p(np.maximum(Pu[:, ou], 0))).mean())
                r.update(design=d, model=m, target=t)
                rows.append(r); preds[(d, m, t)] = (idx, P)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(TAB, 'pooled_metrics_all.csv'), index=False)
    return S, preds


def bootstrap(S, preds, meta, Y):
    ci, pairs = [], []
    for d in DESIGNS:
        unit = meta[A.BOOT_UNIT[d]].values
        for t in TARGETS:
            ms = [m for m in MODELS if (d, m, t) in preds]
            if not ms:
                continue
            idx = preds[(d, ms[0], t)][0]
            ms = [m for m in ms if np.array_equal(preds[(d, m, t)][0], idx)]
            y = Y[t].values[idx]; g = unit[idx]
            ug = np.unique(g); pos = [np.where(g == u)[0] for u in ug]
            rng = np.random.default_rng(7)
            draws = [np.concatenate([pos[i] for i in rng.integers(0, len(ug), len(ug))]) for _ in range(B)]
            bs = {}
            for m in ms:
                P = preds[(d, m, t)][1]
                bs[m] = np.array([r2v(y[k], P[:, k]).mean() for k in draws])
                lo, hi = np.percentile(bs[m], [2.5, 97.5])
                ci.append(dict(design=d, target=t, model=m, R2_lo=lo, R2_hi=hi))
            for i, a in enumerate(ms):
                for b in ms[i + 1:]:
                    dl = bs[a] - bs[b]
                    pairs.append(dict(design=d, target=t, a=a, b=b, d_mean=dl.mean(),
                                      d_lo=np.percentile(dl, 2.5), d_hi=np.percentile(dl, 97.5),
                                      p=min(1.0, 2 * min((dl <= 0).mean(), (dl >= 0).mean()))))
        print('bootstrap', d, flush=True)
    CI = pd.DataFrame(ci); PR = pd.DataFrame(pairs)
    CI.to_csv(os.path.join(TAB, 'boot_ci.csv'), index=False)
    PR.to_csv(os.path.join(TAB, 'boot_pairs.csv'), index=False)
    return CI, PR


def friedman(S):
    out = []
    for d in DESIGNS:
        M = S[S.design == d].pivot(index='target', columns='model', values='R2')
        M = M[[m for m in MODELS if m in M.columns]].dropna()
        if M.shape[1] < 3:
            continue
        rk = M.rank(axis=1, ascending=False)
        chi, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
        k, n = M.shape[1], M.shape[0]
        cd = A.NEMENYI_Q[k] * np.sqrt(k * (k + 1) / (6 * n))
        for c in M.columns:
            out.append(dict(design=d, model=c, mean_rank=rk[c].mean(), chi2=chi, p=p, cd=cd, k=k))
    F = pd.DataFrame(out); F.to_csv(os.path.join(TAB, 'friedman.csv'), index=False)
    return F


def ablation(parts, meta, Y, S):
    var = sorted({r['variant'] for r in parts if r.get('variant') and r['model'] == 'Hybrid'})
    rows = []
    unit = meta.SiteKey.values
    for v, clip in [(v, True) for v in var] + [('head_linear', False)]:
        for d in ['site', 'aez', 'country']:
            Av = A.assemble(parts, d, 'Hybrid', v, clip=clip)
            Af = A.assemble(parts, d, 'Hybrid', '', clip=clip)
            for t in TARGETS:
                if t not in Av or t not in Af:
                    continue
                iv, Pv, _ = Av[t]; i0, P0, _ = Af[t]
                ov, o0 = np.argsort(iv), np.argsort(i0)
                iv, Pv, P0 = iv[ov], Pv[:, ov], P0[:, o0]
                y = Y[t].values[iv]
                rv, r0 = r2v(y, Pv), r2v(y, P0)
                ly = np.log1p(np.maximum(y, 0))
                lrv = r2v(ly, np.log1p(np.maximum(Pv, 0))); lr0 = r2v(ly, np.log1p(np.maximum(P0, 0)))
                g = unit[iv]; ug = np.unique(g); pos = [np.where(g == u)[0] for u in ug]
                rng = np.random.default_rng(3)
                dl = []
                for _ in range(1000):
                    k = np.concatenate([pos[i] for i in rng.integers(0, len(ug), len(ug))])
                    dl.append(r2v(y[k], Pv[:, k]).mean() - r2v(y[k], P0[:, k]).mean())
                rows.append(dict(variant=v, clipped=clip, design=d, target=t, R2_var=rv.mean(), R2_full=r0.mean(),
                                 dR2=rv.mean() - r0.mean(), d_lo=np.percentile(dl, 2.5),
                                 d_hi=np.percentile(dl, 97.5), R2log_var=lrv.mean(), R2log_full=lr0.mean(),
                                 seeds_var=len(Pv)))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(TAB, 'ablation.csv'), index=False)
    return D


def boundedness(parts, D):
    """W_o L1 norms and worst-case comparison hybrid vs linear head."""
    w = [s['wo_l1'] for r in parts if r['model'] == 'Hybrid' and not r.get('variant')
         for s in r['seeds'] if 'wo_l1' in s]
    wl = [s['wo_l1'] for r in parts if r['model'] == 'Hybrid' and r.get('variant') == 'head_linear'
          for s in r['seeds'] if 'wo_l1' in s]
    out = {'wo_l1_median': float(np.median(w)), 'wo_l1_p5': float(np.percentile(w, 5)),
           'wo_l1_p95': float(np.percentile(w, 95)), 'wo_l1_max': float(np.max(w)),
           'wlin_l1_median': float(np.median(wl)) if wl else np.nan}
    if D.empty:
        return out, pd.DataFrame()
    b = D[D.variant == 'head_linear']
    rows = []
    for (d, c), g in b.groupby(['design', 'clipped']):
        rows.append(dict(design=d, clipped=c, worst_rbf=g.R2_full.min(), worst_lin=g.R2_var.min(),
                         p10_rbf=np.percentile(g.R2_full, 10), p10_lin=np.percentile(g.R2_var, 10),
                         worstlog_rbf=g.R2log_full.min(), worstlog_lin=g.R2log_var.min(),
                         mean_rbf=g.R2_full.mean(), mean_lin=g.R2_var.mean(),
                         n_lin_better=int((g.dR2 > 0).sum()), n=len(g)))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(TAB, 'boundedness.csv'), index=False)
    return out, T


def kenya(Y):
    R = pd.DataFrame(exp_rows('exp_kenya'))
    if R.empty:
        return R
    rows = []
    for (t, m, c, r), g in R.groupby(['target', 'model', 'cond', 'rep']):
        if g.test_site.nunique() < 3:
            continue
        idx = np.concatenate(g.idx.values)
        p = np.concatenate([clip_out(Y, t, np.array(i), np.array(v)) for i, v in zip(g.idx.values, g.pred.values)])
        q = metrics(Y[t].values[idx], p)
        rows.append(dict(target=t, model=m, cond=c, rep=r, R2=q['R2'], RMSE=q['RMSE'],
                         n_train=g.n_train.mean(), n_sites=g.n_sites.mean()))
    K = pd.DataFrame(rows).groupby(['target', 'model', 'cond'], as_index=False).mean(numeric_only=True)
    K.to_csv(os.path.join(TAB, 'kenya.csv'), index=False)
    return K


def learning_curves(Y):
    R = pd.DataFrame(exp_rows('exp_lc'))
    if R.empty:
        return R
    rows = []
    for (t, m, ms, n, r), g in R.groupby(['target', 'model', 'm_sites', 'n', 'rep']):
        idx = np.concatenate(g.idx.values)
        p = np.concatenate([np.clip(np.array(v), *A.fold_bounds('site', t, f)) for v, f in zip(g.pred.values, g.fold.values)])
        q = metrics(Y[t].values[idx], p)
        rows.append(dict(target=t, model=m, m_sites=ms, n=n, rep=r, R2=q['R2'], folds=len(g)))
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(TAB, 'learning_curves_raw.csv'), index=False)
    return L


def spiking(Y):
    R = pd.DataFrame(exp_rows('exp_spike'))
    if R.empty:
        return R
    rows = []
    for (t, m, k, r), g in R.groupby(['target', 'model', 'k', 'rep']):
        if len(g) < 8:
            continue
        idx = np.concatenate(g.idx.values)
        p = np.concatenate([np.clip(np.array(v), *A.fold_bounds('site', t, f)) for v, f in zip(g.pred.values, g.fold.values)])
        q = metrics(Y[t].values[idx], p)
        rows.append(dict(target=t, model=m, k=k, rep=r, R2=q['R2'], RMSE=q['RMSE']))
    Sp = pd.DataFrame(rows).groupby(['target', 'model', 'k'], as_index=False).mean(numeric_only=True)
    Sp.to_csv(os.path.join(TAB, 'spiking.csv'), index=False)
    return Sp


def conformal():
    R = pd.DataFrame(exp_rows('exp_conformal'))
    if R.empty:
        return R
    R['nc'] = R.coverage * R.n_test
    rows = []
    for (d, m, c), g in R.groupby(['design', 'model', 'cal']):
        per_t = g.groupby('target').apply(lambda x: x.nc.sum() / x.n_test.sum())
        rows.append(dict(design=d, model=m, cal=c, coverage=g.nc.sum() / g.n_test.sum(),
                         cov_min_target=per_t.min(), cov_max_target=per_t.max(),
                         n_targets_ge85=int((per_t >= 0.85).sum())))
    Cf = pd.DataFrame(rows); Cf.to_csv(os.path.join(TAB, 'conformal.csv'), index=False)
    per = R.groupby(['design', 'model', 'cal', 'target']).apply(
        lambda x: pd.Series({'coverage': x.nc.sum() / x.n_test.sum(), 'width': np.median(x.width_median)})).reset_index()
    per.to_csv(os.path.join(TAB, 'conformal_by_target.csv'), index=False)
    return Cf


def prep():
    R = pd.DataFrame(exp_rows('exp_prep'))
    if R.empty:
        return R
    T = R.pivot_table(index='variant', columns='model', values='R2', aggfunc='mean')
    T.to_csv(os.path.join(TAB, 'prep.csv'))
    return T


THRESH = {'pH': (5.5, 'pH below 5.5'), 'OrgC_pct': (2.0, 'organic C below 2%'),
          'P_Olsen': (15.0, 'Olsen P below 15 mg/kg'), 'K_exch': (0.2, 'exch. K below 0.2 cmolc/kg'),
          'Ca_exch': (2.0, 'exch. Ca below 2 cmolc/kg'), 'Mg_exch': (0.5, 'exch. Mg below 0.5 cmolc/kg'),
          'eCEC': (5.0, 'eCEC below 5 cmolc/kg')}


def screening(preds, Y, model):
    rows = []
    for t, (thr, lab) in THRESH.items():
        if ('site', model, t) not in preds:
            continue
        idx, P = preds[('site', model, t)]
        y = Y[t].values[idx]
        obs = y < thr
        for p in [P.mean(0)]:
            pr = p < thr
            tp = (obs & pr).sum(); fn = (obs & ~pr).sum(); tn = (~obs & ~pr).sum(); fp = (~obs & pr).sum()
            se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
            rows.append(dict(model=model, target=t, rule=lab, prevalence=obs.mean(), sens=se, spec=sp,
                             bal_acc=(se + sp) / 2, n=len(y)))
    return pd.DataFrame(rows)


def p_ceiling(Y):
    y = Y['P_Olsen'].values; y = y[np.isfinite(y)]
    out = {}
    for cv in [0.1, 0.2, 0.3]:
        for floor in [0.5, 1.0]:
            err = (cv * y) ** 2 + floor ** 2
            out[f'{int(cv*100)}_{floor}'] = dict(orig=1 - err.mean() / y.var(),
                                                 log=1 - np.mean((np.log1p(y + np.sqrt(err)) - np.log1p(y)) ** 2) / np.log1p(y).var())
    return out


def main():
    X, wnb, meta, Y = load()
    parts = A.load_parts()
    print(len(parts), 'parts', flush=True)
    S, preds = main_tables(parts, X, meta, Y)
    CI, PR = bootstrap(S, preds, meta, Y)
    F = friedman(S)
    D = ablation(parts, meta, Y, S)
    bnd, BT = boundedness(parts, D)
    K = kenya(Y); L = learning_curves(Y); Sp = spiking(Y); Cf = conformal(); Pp = prep()
    scr = pd.concat([screening(preds, Y, m) for m in MODELS])
    scr.to_csv(os.path.join(TAB, 'screening.csv'), index=False)
    json.dump({'bound': bnd, 'p_ceiling': p_ceiling(Y)}, open(os.path.join(TAB, 'misc.json'), 'w'), indent=1)
    import pickle
    pickle.dump(preds, open(os.path.join(OUT, 'preds.pkl'), 'wb'))
    print(S.pivot_table(index='model', columns='design', values='R2', aggfunc='mean').round(3).loc[[m for m in MODELS if m in S.model.unique()], [d for d in DESIGNS if d in S.design.unique()]])
    print(F.pivot(index='model', columns='design', values='mean_rank').round(2))
    if not D.empty:
        print(D[D.clipped].groupby(['design', 'variant']).agg(dR2=('dR2', 'mean'), n_better=('dR2', lambda x: (x > 0).sum()), n=('dR2', 'size')).round(3))
    print(BT); print(bnd)


if __name__ == '__main__':
    main()
