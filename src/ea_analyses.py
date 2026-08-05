"""
kenya_analyses.py -- Aggregate the benchmark and run the supporting analyses
that the manuscript reports. Everything is written to kenya/.

  aggregate   per-model x per-target metrics with fold SD and 95% CI
  attention   DAFM attention over MIR wavenumbers (input-level, interpretable)
  gate        branch gate: how much weight goes to spectra vs ancillary covariates
  deficiency  agronomic status against indicative critical levels
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ea as R                                    # noqa: E402
from ml_dual import DualBranchDAFM                       # noqa: E402
from cv import metrics                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KEN = os.path.join(HERE, 'ea')
MODELS = ['PLSR', 'Ridge', 'RandomForest', 'GradientBoosting', 'MLP', 'DualDAFM']
LABEL = {'PLSR': 'PLSR', 'Ridge': 'Ridge', 'RandomForest': 'RF',
         'GradientBoosting': 'GBM', 'MLP': 'MLP', 'DualDAFM': 'Hybrid'}
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

# Indicative critical levels for tropical arable topsoil. Reported with a
# +/-50% sensitivity band in the manuscript, not as point truth.
CRIT = {
    'pH': (5.5, 'below this, Al toxicity risk rises'),
    'OrgC_pct': (2.0, 'organic C, % by weight'),
    'TotN_pct': (0.2, 'total N, % by weight'),
    'P_Olsen': (15.0, 'Olsen-extractable P, mg/kg'),
    'Ca_exch': (2.0, 'exchangeable Ca, cmolc/kg'),
    'Mg_exch': (0.5, 'exchangeable Mg, cmolc/kg'),
    'K_exch': (0.2, 'exchangeable K, cmolc/kg'),
    'eCEC': (5.0, 'effective CEC, cmolc/kg'),
    'Zn_total': (50.0, 'total Zn, mg/kg'),
    'Cu_total': (20.0, 'total Cu, mg/kg'),
}


def summarise(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan, k=0)
    mu = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    h = TCRIT.get(len(v) - 1, 2.776) * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return dict(mean=mu, sd=sd, lo=mu - h, hi=mu + h, k=int(len(v)))


def run_aggregate():
    rows = []
    for f in glob.glob(os.path.join(KEN, 'parts', '*__f*.json')):
        d = json.load(open(f))
        if 'metrics' not in d:
            continue
        de, t, m, ph = d['task']
        rows.append(dict(design=de, target=t, model=m, fold=int(ph[1:]),
                         **{k: d['metrics'][k] for k in
                            ['R2', 'RMSE', 'MAE', 'bias', 'nRMSE_pct']}))
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(KEN, 'fold_metrics.csv'), index=False)

    out = []
    for (de, t, m), g in raw.groupby(['design', 'target', 'model']):
        rec = dict(design=de, target=t, model=m)
        for k in ['R2', 'RMSE', 'MAE', 'bias', 'nRMSE_pct']:
            s = summarise(g[k].values)
            rec[k] = s['mean']
            rec[k + '_sd'] = s['sd']
            rec[k + '_lo'] = s['lo']
            rec[k + '_hi'] = s['hi']
        rec['n_folds'] = len(g)
        out.append(rec)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(KEN, 'summary.csv'), index=False)

    # headline robustness statistics used in the abstract and conclusion
    stats = {}
    for de in ['cluster', 'site', 'country']:
        p = res[res.design == de].pivot(index='target', columns='model', values='R2')[MODELS]
        rk = p.rank(axis=1, ascending=False)
        stats[de] = {
            'mean_R2': {m: round(float(p[m].mean()), 4) for m in MODELS},
            'median_R2': {m: round(float(p[m].median()), 4) for m in MODELS},
            'mean_rank': {m: round(float(rk[m].mean()), 3) for m in MODELS},
            'n_positive': {m: int((p[m] > 0).sum()) for m in MODELS},
            'worst_R2': {m: round(float(p[m].min()), 3) for m in MODELS},
            'n_targets': int(len(p)),
            'best_by_mean_rank': str(rk.mean().idxmin()),
        }
    json.dump(stats, open(os.path.join(KEN, 'headline.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1))


def run_attention():
    """DAFM attention over wavenumbers, plus the spectral/covariate branch gate.

    Attention is defined over convolutional output positions; each position maps
    back to a contiguous wavenumber window, which is reported so the weights are
    attributable to interpretable absorbance regions.
    """
    X, wn, meta, Y = R.load()
    res = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    tg = ['TotN_pct', 'OrgC_pct', 'pH', 'Ca_exch']
    cache = os.path.join(KEN, '_attn')
    os.makedirs(cache, exist_ok=True)
    rows, gates = [], []
    for t in tg:
        cp = os.path.join(cache, f'{t}.json')
        if os.path.exists(cp):
            d = json.load(open(cp))
            rows.extend(d['rows']); gates.append(d['gate'])
            continue
        y = Y[t].values.astype(float)
        ok = np.isfinite(y)
        folds = R.folds_for('cluster', meta)
        params = json.load(open(os.path.join(
            KEN, 'parts', f'cluster__{t}__DualDAFM__f1.json')))['params']
        log = bool(pd.Series(y[ok]).skew() > 2.0)
        A, G = [], []
        for f in range(1, 5):
            fp = os.path.join(cache, f'{t}__f{f}.json')
            if os.path.exists(fp):
                d = json.load(open(fp))
                A.append(d['a']); G.append(d['g'])
                continue
            te = np.array([i for i in folds[f] if ok[i]])
            tr = np.array([i for i in range(len(y)) if ok[i] and i not in set(folds[f])])
            sp = R.SpecPrep(**R.SPEC_CFG).fit(X[tr])
            yp = R.YPrep(log).fit(y[tr])
            Xtr, Xte = sp.transform(X[tr]), sp.transform(X[te])
            Ctr, Cte = R.covariates(meta, tr), R.covariates(meta, te)
            cm, cs = Ctr.mean(0), Ctr.std(0)
            cs[cs < 1e-12] = 1.0
            Ctr, Cte = (Ctr - cm) / cs, (Cte - cm) / cs
            m = DualBranchDAFM(Xtr.shape[1], Ctr.shape[1], seed=R.SEED, **params)
            m.fit(Xtr, Ctr, yp.fwd(y[tr]), Xte, Cte, yp.fwd(y[te]))
            a_ = m.spectral_attention(Xte, Cte).tolist()
            g_ = m.branch_gate(Xte, Cte).tolist()
            json.dump({'a': a_, 'g': g_}, open(fp, 'w'))
            A.append(a_); G.append(g_)
            print(f'  {t} fold {f} done', flush=True)
        if len(A) < 4:
            continue
        A = np.array(A)
        G = np.array(G)
        # map attention positions back to wavenumber centres
        binned = R.bin_spectra(wn[None, :], R.SPEC_CFG['binf'])[0]
        S2 = A.shape[1]
        step = len(binned) / S2
        centres = [float(binned[int((i + 0.5) * step)]) for i in range(S2)]
        rws = [dict(target=t, position=i, wavenumber_cm1=centres[i],
                    attention=float(A[:, i].mean()),
                    sd=float(A[:, i].std(ddof=1))) for i in range(S2)]
        gt = dict(target=t,
                  spectral_weight=float(G[:, 0].mean()),
                  spectral_sd=float(G[:, 0].std(ddof=1)),
                  covariate_weight=float(G[:, 1].mean()))
        json.dump({'rows': rws, 'gate': gt}, open(cp, 'w'))
        rows.extend(rws); gates.append(gt)
        print('attention done', t, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(KEN, 'attention.csv'), index=False)
    pd.DataFrame(gates).to_csv(os.path.join(KEN, 'branch_gate.csv'), index=False)
    print(pd.DataFrame(gates).round(3).to_string(index=False))


def run_deficiency():
    _, _, meta, Y = R.load()
    top = meta.Depth == 'Topsoil'
    rows = []
    for k, (c, basis) in CRIT.items():
        v = Y[k].values.astype(float)[top.values]
        v = v[np.isfinite(v)]
        rows.append(dict(target=k, critical=c, basis=basis, n=len(v),
                         pct_below=round(100.0 * float((v < c).mean()), 1),
                         pct_below_half=round(100.0 * float((v < 0.5 * c).mean()), 1),
                         pct_below_1p5=round(100.0 * float((v < 1.5 * c).mean()), 1),
                         median=round(float(np.median(v)), 3)))
    d = pd.DataFrame(rows).sort_values('pct_below', ascending=False)
    d.to_csv(os.path.join(KEN, 'deficiency.csv'), index=False)
    print(d.to_string(index=False))


if __name__ == '__main__':
    {'aggregate': run_aggregate, 'attention': run_attention,
     'deficiency': run_deficiency}[sys.argv[1]]()
