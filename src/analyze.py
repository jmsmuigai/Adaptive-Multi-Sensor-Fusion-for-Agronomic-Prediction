"""
analyze.py -- aggregate out/parts into pooled out-of-fold metrics, group
bootstrap CIs, paired group-bootstrap tests, Friedman/Nemenyi ranks.

Primary statistic: metrics on POOLED out-of-fold predictions (every sample is
predicted exactly once per design). For neural models every seed is scored
separately and the seed mean (and SD) is reported; bootstrap resamples apply
the same group draw to every seed and average, so seed variance and sampling
variance are both carried into the interval.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, LABEL, OUT, NFOLDS, outer_folds, metrics, YPrep

B = 2000
RNG = np.random.default_rng(7)
MODELS_ORDER = ['PLSR', 'Ridge', 'SVR', 'KRR', 'GPR', 'Cubist', 'MBL', 'RF', 'LightGBM',
                'MLP', 'CNN', 'MOCNN', 'Hybrid']
NEMENYI_Q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
             9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}
BOOT_UNIT = {'random': 'ClusterKey', 'cluster': 'ClusterKey', 'site': 'SiteKey',
             'country': 'SiteKey', 'aez': 'SiteKey'}


def load_parts(pattern='*'):
    rows = []
    for p in glob.glob(os.path.join(OUT, 'parts', pattern + '.json')):
        rows.append(json.load(open(p)))
    return rows


_BOUNDS = {}


def fold_bounds(design, t, f):
    """min/max of the target in the training rows of an outer fold."""
    k = (design, t, f)
    if k not in _BOUNDS:
        X, wnb, meta, Y = load()
        y = Y[t].values
        te = outer_folds(design, meta)[f]
        tr = np.setdiff1d(np.where(np.isfinite(y))[0], te)
        _BOUNDS[k] = (float(np.min(y[tr])), float(np.max(y[tr])))
    return _BOUNDS[k]


CLIP = True


def assemble(parts, design, model, variant='', clip=None):
    """Return dict target -> (idx, preds[S, n], fold_of) pooled over folds.
    With clip=True (default), back-transformed predictions are limited to the
    range of the target in the training fold -- a safeguard applied
    identically to every model (see manuscript Section 2.8)."""
    clip = CLIP if clip is None else clip
    X, wnb, meta, Y = load()
    out = {}
    for t in TARGETS:
        P = [r for r in parts if r['design'] == design and r['model'] == model
             and r.get('variant', '') == variant
             and (r['target'] == t or (model == 'MOCNN' and r['target'] == 'ALL'))]
        if len(P) < NFOLDS[design]:
            continue
        idx, pr, fo = [], [], []
        for r in sorted(P, key=lambda r: r['fold']):
            ii = np.array(r['test_idx'])
            if model == 'MOCNN':
                v = np.array(r['pred_by_target'][t])
                ok = np.isfinite(Y[t].values[ii])
                ii, v = ii[ok], v[:, ok]
            else:
                v = np.array(r['pred'])
                if v.ndim == 1:
                    v = v[None]
            if clip:
                lo, hi = fold_bounds(design, t, r['fold'])
                v = np.clip(v, lo, hi)
            idx.append(ii); pr.append(v); fo.append(np.full(len(ii), r['fold']))
        out[t] = (np.concatenate(idx), np.concatenate(pr, 1), np.concatenate(fo))
    return out


def train_stats(design, t):
    """Per-fold training mean / SD of the target (for MAE skill, RMSE/SD)."""
    X, wnb, meta, Y = load()
    y = Y[t].values
    folds = outer_folds(design, meta)
    ok = np.isfinite(y)
    m, s = {}, {}
    for f, te in enumerate(folds):
        tr = np.setdiff1d(np.where(ok)[0], te)
        m[f], s[f] = float(np.mean(y[tr])), float(np.std(y[tr], ddof=1))
    return m, s


def pooled_metrics(y, P, fo, tm, ts):
    """P: (S, n). Returns dict of seed-mean metrics and seed SD of R2."""
    res = []
    base = np.array([tm[f] for f in fo])
    sdt = np.mean([ts[f] for f in np.unique(fo)])
    mae0 = np.mean(np.abs(y - base))
    for p in P:
        m = metrics(y, p)
        m['nRMSE_sdtrain'] = m['RMSE'] / sdt
        m['MAE_skill'] = 1 - m['MAE'] / mae0
        ly, lp = np.log1p(np.maximum(y, 0)), np.log1p(np.maximum(p, 0))
        m['R2_log'] = 1 - np.sum((lp - ly) ** 2) / np.sum((ly - ly.mean()) ** 2)
        res.append(m)
    df = pd.DataFrame(res)
    out = df.mean().to_dict()
    out['R2_seed_sd'] = float(df.R2.std(ddof=1)) if len(df) > 1 else 0.0
    out['n_seeds'] = len(df)
    # per-fold R2 (seed mean) for the dispersion summary
    fr = []
    for f in np.unique(fo):
        k = fo == f
        if k.sum() > 2 and np.var(y[k]) > 0:
            fr.append(np.mean([1 - np.sum((p[k] - y[k]) ** 2) / np.sum((y[k] - y[k].mean()) ** 2) for p in P]))
    out['fold_R2_min'] = float(np.min(fr)) if fr else np.nan
    out['fold_R2_median'] = float(np.median(fr)) if fr else np.nan
    out['fold_R2_max'] = float(np.max(fr)) if fr else np.nan
    return out


def boot_draws(groups, B=B, seed=7):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in ug}
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True)
        yield np.concatenate([pos[g] for g in pick])


def r2(y, p):
    return 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    X, wnb, meta, Y = load()
    parts = load_parts()
    print(len(parts), 'parts')
    rows, preds = [], {}
    for d in NFOLDS:
        for m in MODELS_ORDER:
            A = assemble(parts, d, m)
            for t, (idx, P, fo) in A.items():
                tm, ts = train_stats(d, t)
                y = Y[t].values[idx]
                r = pooled_metrics(y, P, fo, tm, ts)
                r.update(design=d, model=m, target=t)
                rows.append(r)
                preds[(d, m, t)] = (idx, P)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, 'pooled_metrics.csv'), index=False)

    # group bootstrap CIs and paired tests (same draws for all models)
    ci_rows, pair_rows = [], []
    for d in NFOLDS:
        unit = meta[BOOT_UNIT[d]].values
        for t in TARGETS:
            ms = [m for m in MODELS_ORDER if (d, m, t) in preds]
            if not ms:
                continue
            idx0 = np.sort(preds[(d, ms[0], t)][0])
            # align every model on the common index set
            al = {}
            for m in ms:
                idx, P = preds[(d, m, t)]
                o = np.argsort(idx)
                if not np.array_equal(idx[o], idx0):
                    common_ = np.intersect1d(idx0, idx)
                    idx0 = common_
                al[m] = (idx[o], P[:, o])
            al = {m: (i, P[:, np.isin(i, idx0)]) for m, (i, P) in al.items()}
            y = Y[t].values[idx0]
            g = unit[idx0]
            draws = list(boot_draws(g, B))
            bs = {m: np.array([np.mean([r2(y[k], p[k]) for p in al[m][1]]) for k in draws]) for m in ms}
            for m in ms:
                lo, hi = np.percentile(bs[m], [2.5, 97.5])
                ci_rows.append(dict(design=d, target=t, model=m, R2_lo=lo, R2_hi=hi))
            for a in ms:
                for b in ms:
                    if a >= b:
                        continue
                    dlt = bs[a] - bs[b]
                    p = 2 * min((dlt <= 0).mean(), (dlt >= 0).mean())
                    pair_rows.append(dict(design=d, target=t, a=a, b=b,
                                          d_lo=np.percentile(dlt, 2.5), d_hi=np.percentile(dlt, 97.5),
                                          d_mean=dlt.mean(), p=min(1.0, p)))
        print('bootstrap done', d, flush=True)
    CI = pd.DataFrame(ci_rows); CI.to_csv(os.path.join(OUT, 'boot_ci.csv'), index=False)
    PR = pd.DataFrame(pair_rows); PR.to_csv(os.path.join(OUT, 'boot_pairs.csv'), index=False)

    # Friedman / Nemenyi on R2 across the 13 properties
    fr_rows = []
    for d in NFOLDS:
        M = S[S.design == d].pivot(index='target', columns='model', values='R2')
        M = M[[m for m in MODELS_ORDER if m in M.columns]].dropna()
        if M.shape[1] < 3 or M.shape[0] < 3:
            continue
        ranks = M.rank(axis=1, ascending=False)
        chi, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
        k, N = M.shape[1], M.shape[0]
        cd = NEMENYI_Q[k] * np.sqrt(k * (k + 1) / (6 * N))
        for c in M.columns:
            fr_rows.append(dict(design=d, model=c, mean_rank=ranks[c].mean(),
                                friedman_chi2=chi, friedman_p=p, nemenyi_cd=cd, k=k, N=N))
    FR = pd.DataFrame(fr_rows); FR.to_csv(os.path.join(OUT, 'friedman.csv'), index=False)
    print(FR.pivot(index='model', columns='design', values='mean_rank').round(2))
    print(S.pivot_table(index='model', columns='design', values='R2', aggfunc='mean').round(3))


if __name__ == '__main__':
    main()
