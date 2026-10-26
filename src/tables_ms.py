"""tables_ms.py -- manuscript and supplementary tables as Markdown (pandoc pipe tables)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, LABEL, OUT, outer_folds, AEZ_NAME, SITE_AEZ, NFOLDS
import run_main as RM
from classical import MODELS as CM

T = os.path.join(OUT, 'tables'); MT = os.path.join(OUT, 'ms_tables'); os.makedirs(MT, exist_ok=True)
UNITS = {'pH': '–', 'OrgC_pct': '%', 'TotN_pct': '%', 'P_Olsen': 'mg kg⁻¹', 'Ca_exch': 'cmolc kg⁻¹',
         'Mg_exch': 'cmolc kg⁻¹', 'K_exch': 'cmolc kg⁻¹', 'eCEC': 'cmolc kg⁻¹', 'S_total': 'mg kg⁻¹',
         'Fe_total': 'mg kg⁻¹', 'Mn_total': 'mg kg⁻¹', 'Zn_total': 'mg kg⁻¹', 'Cu_total': 'mg kg⁻¹'}
MODELS = ['PLSR', 'Ridge', 'SVR', 'KRR', 'GPR', 'Cubist', 'MBL', 'RF', 'LightGBM', 'MLP', 'CNN', 'MOCNN', 'Hybrid']
MNAME = {'LightGBM': 'LightGBM', 'MOCNN': 'MO-CNN', 'RF': 'RF'}


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    s = f'{x:.{d}f}'
    return s.replace('-', '−')


def md(df):
    cols = list(df.columns)
    out = '| ' + ' | '.join(str(c) for c in cols) + ' |\n'
    al = ['---' if i == 0 else '---:' for i in range(len(cols))]
    out += '| ' + ' | '.join(al) + ' |\n'
    for _, r in df.iterrows():
        out += '| ' + ' | '.join(str(v) for v in r.values) + ' |\n'
    return out


def save(name, df):
    txt = md(df).replace(' (round 1)', '').replace('round-1 Gaussian activation', 'fixed-centre Gaussian activation')
    open(os.path.join(MT, name + '.md'), 'w').write(txt)


def main():
    X, wnb, meta, Y = load()
    S = pd.read_csv(os.path.join(T, 'pooled_metrics_all.csv'))
    CI = pd.read_csv(os.path.join(T, 'boot_ci.csv'))
    F = pd.read_csv(os.path.join(T, 'friedman.csv'))

    # Table 1
    rows = []
    for c in ['Kenya', 'Ethiopia', 'Tanzania', 'Uganda']:
        m = meta[meta.Country == c]
        z = ', '.join(sorted({SITE_AEZ[s] for s in m.SiteKey.unique()}, key=['Aw', 'Am', 'BSh', 'BWh', 'C'].index))
        rows.append([c, len(m), m.SiteKey.nunique(), m.ClusterKey.nunique(),
                     f'{m.Latitude.min():.2f} to {m.Latitude.max():.2f}'.replace('-', '−'),
                     f'{m.Longitude.min():.2f} to {m.Longitude.max():.2f}', z])
    rows.append(['**Total**', f'**{len(meta)}**', f'**{meta.SiteKey.nunique()}**', f'**{meta.ClusterKey.nunique()}**', '–', '–', '–'])
    save('T1', pd.DataFrame(rows, columns=['Country', 'Samples', 'Sentinel sites', 'LDSF clusters',
                                           'Latitude range (°)', 'Longitude range (°)', 'Köppen–Geiger groups']))
    # Table 2
    src = {'pH': ('pH (water, 1:2.5)', 'Potentiometric', 'Yes'), 'OrgC_pct': ('Percent_Org_C', 'Dry combustion', 'Yes'),
           'TotN_pct': ('Percent_N', 'Dry combustion', 'No'), 'P_Olsen': ('Olsen.P', 'NaHCO₃ extraction', 'Yes'),
           'Ca_exch': ('Caex', 'Cobalt hexamine', 'Yes'), 'Mg_exch': ('Mgex', 'Cobalt hexamine', 'Yes'),
           'K_exch': ('Kex', 'Cobalt hexamine', 'Yes'), 'eCEC': ('eCEC', 'Sum of cations', 'Yes'),
           'S_total': ('S', 'Aqua regia', 'No'), 'Fe_total': ('Fe', 'Aqua regia', 'No'),
           'Mn_total': ('Mn', 'Aqua regia', 'No'), 'Zn_total': ('Zn', 'Aqua regia', 'No'), 'Cu_total': ('Cu', 'Aqua regia', 'No')}
    rows = []
    for t in TARGETS:
        y = Y[t].dropna()
        rows.append([LABEL[t], src[t][0], src[t][1], UNITS[t], int(y.size), f'{y.median():.3g}', f(y.skew(), 2),
                     'log(1+y)' if y.skew() > 2 else 'none', src[t][2]])
    save('T2', pd.DataFrame(rows, columns=['Property', 'Source column', 'Method', 'Unit', 'n', 'Median', 'Skew',
                                           'Transform', 'Interpretive thresholds used?']))
    # Table 3 design effect
    M = S.groupby(['model', 'design']).R2.mean().unstack()
    Fr = F[F.design == 'site'].set_index('model').mean_rank
    rows = []
    for m in MODELS:
        r = [MNAME.get(m, m)] + [f(M.loc[m].get(d, np.nan)) for d in ['random', 'cluster', 'site', 'aez', 'country']]
        r.append(f(Fr.get(m, np.nan), 2))
        rows.append(r)
    best = {d: M[d].max() for d in M.columns}
    for r in rows:
        for j, d in enumerate(['random', 'cluster', 'site', 'aez', 'country']):
            if r[j + 1] == f(best[d]):
                r[j + 1] = f'**{r[j + 1]}**'
    save('T3', pd.DataFrame(rows, columns=['Model', 'Random', 'Leave-cluster-out', 'Leave-site-out', 'Leave-AEZ-out',
                                           'Leave-country-out', 'Mean rank (site)']))
    # Table 4 site per property (GPR = best mean rank; per-property maximum shown only descriptively)
    site = S[S.design == 'site']
    rows = []
    for t in site[site.model == 'GPR'].sort_values('R2', ascending=False).target:
        g = site[site.target == t].set_index('model')
        b = g.R2.idxmax()
        ci = CI[(CI.design == 'site') & (CI.target == t)].set_index('model')
        rows.append([LABEL[t], f'{f(g.R2["GPR"])} ({f(ci.R2_lo["GPR"], 2)}, {f(ci.R2_hi["GPR"], 2)})',
                     f(g.RMSE['GPR'], 3 if g.RMSE['GPR'] < 10 else 0), f(g.RPD['GPR'], 2), f(g.RPIQ['GPR'], 2),
                     f(g.CCC['GPR'], 2),
                     f(g.R2['PLSR']), f(g.R2['CNN']),
                     f'{f(g.R2["Hybrid"])} ({f(ci.R2_lo["Hybrid"], 2)}, {f(ci.R2_hi["Hybrid"], 2)})',
                     f'{f(g.R2[b])} ({MNAME.get(b, b)})'])
    save('T4', pd.DataFrame(rows, columns=['Property', 'GPR R² (95% CI)', 'GPR RMSE', 'RPD', 'RPIQ', 'CCC',
                                           'PLSR R²', 'CNN R²', 'Hybrid R² (95% CI)', 'Highest of 13 models*']))
    # Table 5 ablation
    D = pd.read_csv(os.path.join(T, 'ablation.csv')); D = D[(D.design == 'site') & D.clipped]
    lab = [('head_linear', 'Head', 'RBF head → linear output'), ('head_mlp', 'Head', 'RBF head → dense ReLU layer'),
           ('head_gauss', 'Head', 'RBF head → fixed-centre Gaussian activation'),
           ('pool_mean', 'Pooling', 'Attention pooling → mean pooling'),
           ('pool_reweight', 'Pooling', 'Attention pooling → attention re-weighting + flatten'),
           ('no_pos', 'Pooling', 'Positional embedding removed'),
           ('fuse_concat', 'Fusion', 'Gate → plain concatenation'),
           ('spectra_only', 'Fusion', 'Ancillary branch removed (spectra only)'),
           ('plus_coords', 'Inputs', 'Latitude and longitude added')]
    rows = []
    for v, grp, desc in lab:
        g = D[D.variant == v]
        rows.append([grp, desc, f(g.R2_var.mean()), f(g.dR2.mean()), f'{int((g.dR2 > 0).sum())}/13',
                     int((g.d_lo > 0).sum()), int((g.d_hi < 0).sum())])
    full = D.groupby('target').R2_full.first().mean()
    rows.insert(0, ['–', 'Full hybrid (reference)', f(full), '–', '–', '–', '–'])
    save('T5', pd.DataFrame(rows, columns=['Component', 'Change', 'Mean R²', 'Mean ΔR²', 'Properties improved',
                                           'Significant gains', 'Significant losses']))
    # Table 6 boundedness
    B = pd.read_csv(os.path.join(T, 'boundedness.csv'))
    rows = []
    for d in ['site', 'aez', 'country']:
        for c in [False, True]:
            r = B[(B.design == d) & (B.clipped == c)].iloc[0]
            rows.append([{'site': 'Leave-site-out', 'aez': 'Leave-AEZ-out', 'country': 'Leave-country-out'}[d],
                         'clipped' if c else 'raw', f(r.worst_rbf), f(r.worst_lin), f(r.p10_rbf), f(r.p10_lin),
                         f(r.worstlog_rbf), f(r.worstlog_lin), f(r.mean_rbf), f(r.mean_lin)])
    save('T6', pd.DataFrame(rows, columns=['Design', 'Back-transform', 'Worst R² RBF', 'Worst R² linear',
                                           '10th pct RBF', '10th pct linear', 'Worst log-R² RBF', 'Worst log-R² linear',
                                           'Mean R² RBF', 'Mean R² linear']))
    # Table 7 kenya
    K = pd.read_csv(os.path.join(T, 'kenya.csv')); K = K[K.model.isin(['PLSR', 'KRR', 'Cubist', 'Hybrid'])]
    Km = K.groupby(['model', 'cond']).R2.mean().unstack()
    nn = K.groupby('cond')[['n_train', 'n_sites']].mean()
    rows = []
    for c, lb in [('K2', 'Two Kenyan sites (K2)'), ('K2+EA_n', 'K2 + EA, size-matched to K2'),
                  ('EA', 'Other countries only (EA)'), ('K2+EA', 'K2 + EA (full)')]:
        rows.append([lb, int(round(nn.loc[c, 'n_train'])), f(nn.loc[c, 'n_sites'], 1)] +
                    [f(Km.loc[m, c]) for m in ['PLSR', 'KRR', 'Cubist', 'Hybrid']] + [f(Km[c].mean())])
    save('T7', pd.DataFrame(rows, columns=['Calibration set', 'Training n', 'Training sites', 'PLSR', 'KRR',
                                           'Cubist', 'Hybrid', 'Mean']))
    # Table 8 spiking
    Sp = pd.read_csv(os.path.join(T, 'spiking.csv'))
    rows = []
    for m in ['PLSR', 'KRR', 'LightGBM', 'Hybrid']:
        g = Sp[Sp.model == m]
        rows.append([MNAME.get(m, m), 'Mean of 13'] + [f(g[g.k == k].R2.mean()) for k in [0, 5, 10, 20, 50]])
    for t in ['pH', 'OrgC_pct', 'Fe_total', 'Cu_total', 'K_exch', 'P_Olsen']:
        g = Sp[(Sp.model == 'KRR') & (Sp.target == t)].groupby('k').R2.mean()
        rows.append(['KRR', LABEL[t]] + [f(g.get(k, np.nan)) for k in [0, 5, 10, 20, 50]])
    save('T8', pd.DataFrame(rows, columns=['Model', 'Property', 'k = 0', 'k = 5', 'k = 10', 'k = 20', 'k = 50']))
    # Table 9 conformal
    C = pd.read_csv(os.path.join(T, 'conformal.csv'))
    rows = []
    for d in ['site', 'aez', 'country']:
        for m in ['KRR', 'LightGBM', 'Hybrid']:
            g = C[(C.design == d) & (C.model == m)].set_index('cal')
            rows.append([{'site': 'Leave-site-out', 'aez': 'Leave-AEZ-out', 'country': 'Leave-country-out'}[d],
                         MNAME.get(m, m), f(g.coverage['random']), f(g.coverage['group']),
                         f'{f(g.cov_min_target["group"], 2)}–{f(g.cov_max_target["group"], 2)}'])
    save('T9', pd.DataFrame(rows, columns=['Design', 'Model', 'Coverage, random calibration',
                                           'Coverage, group calibration', 'Range over properties (group)']))
    # Table 10 screening
    sc = pd.read_csv(os.path.join(T, 'screening.csv'))
    nums = json.load(open(os.path.join(OUT, 'numbers.json')))
    best = nums['best_site']
    rows = []
    for t in ['OrgC_pct', 'pH', 'eCEC', 'Ca_exch', 'Mg_exch', 'K_exch', 'P_Olsen']:
        r = sc[(sc.model == best) & (sc.target == t)].iloc[0]
        h = sc[(sc.model == 'Hybrid') & (sc.target == t)].iloc[0]
        rows.append([LABEL[t], r.rule.split(' below ')[1], f(100 * r.prevalence, 0), f(r.sens, 2), f(r.spec, 2),
                     f(r.bal_acc, 2), f(h.bal_acc, 2)])
    save('T10', pd.DataFrame(rows, columns=['Property', 'Critical level (below)', 'Samples below (%)',
                                            f'Sensitivity ({best})', f'Specificity ({best})',
                                            f'Balanced accuracy ({best})', 'Balanced accuracy (Hybrid)']))
    # ------------------------------------------------ supplementary
    rows = []
    for d in ['cluster', 'site', 'aez', 'country']:
        for k, idx in enumerate(outer_folds(d, meta)):
            m = meta.iloc[idx]
            if d == 'cluster':
                desc = f'{m.ClusterKey.nunique()} clusters from {m.SiteKey.nunique()} sites'
            elif d == 'site':
                desc = '; '.join(f'{s.split("_")[1]} ({s.split("_")[0]}, n={int((m.SiteKey == s).sum())})' for s in sorted(m.SiteKey.unique()))
            elif d == 'aez':
                desc = AEZ_NAME[m.AEZ.iloc[0]] + ': ' + ', '.join(s.split('_')[1] for s in sorted(m.SiteKey.unique()))
            else:
                desc = m.Country.iloc[0]
            rows.append([{'cluster': 'Leave-cluster-out', 'site': 'Leave-site-out', 'aez': 'Leave-AEZ-out', 'country': 'Leave-country-out'}[d],
                         k + 1, len(idx), desc])
    rows.append(['Random', '1–5', '262 each', 'random partition (seed 20260804)'])
    save('S1', pd.DataFrame(rows, columns=['Design', 'Fold', 'Test n', 'Withheld units']))
    rows = []
    for m in ['PLSR', 'Ridge', 'SVR', 'KRR', 'GPR', 'Cubist', 'MBL', 'RF', 'LightGBM']:
        mk, g, s = CM[m]
        rows.append([m, {'kfold': '3 group-disjoint inner folds', 'split': 'one group-disjoint inner split',
                         'none': 'none (fixed settings; GPR kernel by marginal likelihood)'}[s],
                     '; '.join(', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in p.items()) for p in g) or '–'])
    for m in ['MLP', 'CNN', 'MOCNN', 'Hybrid']:
        rows.append([m, 'one group-disjoint inner split; early stopping on the same split',
                     '; '.join('{' + ', '.join(f'{k}={v}' for k, v in p.items() if k in ('lr', 'c1', 'c2', 'k1', 's1', 'h', 'h1', 'h2', 'R', 'dropout', 'wd')) + '}' for p in RM.SPACE[m])])
    save('S2', pd.DataFrame(rows, columns=['Model', 'Selection', 'Candidate configurations']))
    rows = []
    for d in ['random', 'cluster', 'site', 'aez', 'country']:
        for t in TARGETS:
            g = S[(S.design == d) & (S.target == t)].set_index('model')
            rows.append([d, LABEL[t]] + [f(g.R2.get(m, np.nan), 2) for m in MODELS])
    save('S3', pd.DataFrame(rows, columns=['Design', 'Property'] + [MNAME.get(m, m) for m in MODELS]))
    ag = pd.read_csv(os.path.join(OUT, 'agreement.csv'))
    rows = []
    for (d, m), g in ag.groupby(['design', 'model']):
        rows.append([d, m, len(g), f(g.R2.clip(-3).mean()), f(g.R2_lib.clip(-3).mean()),
                     f(np.median(np.abs(g.R2 - g.R2_lib))), f(np.corrcoef(g.R2.clip(-3), g.R2_lib.clip(-3))[0, 1], 2)])
    save('S4', pd.DataFrame(rows, columns=['Design', 'Model', 'Fold × property pairs', 'Mean fold R² (round-1 NumPy)',
                                           'Mean fold R² (library)', 'Median |difference|', 'Correlation']))
    P = pd.read_csv(os.path.join(T, 'prep.csv'))
    save('S5', P.assign(KRR=P.KRR.map(f), PLSR=P.PLSR.map(f)).rename(columns={'variant': 'Preprocessing'}))
    print('tables written', sorted(os.listdir(MT)))

def supp_extra():
    import glob, json
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'exp', 'exp_interp', '*.json'))]
    sp = pd.DataFrame([dict(target=r['target'], **r['spearman']) for r in rows]).groupby('target').mean()
    sp.index = [LABEL[t] for t in sp.index]
    cols = {'attention~occlusion': 'Attention–occlusion', 'attention~IG': 'Attention–IG', 'attention~VIP': 'Attention–VIP',
            'occlusion~IG': 'Occlusion–IG', 'occlusion~VIP': 'Occlusion–VIP', 'IG~VIP': 'IG–VIP'}
    sp = sp.rename(columns=cols)[list(cols.values())]
    save('S6', pd.concat([pd.Series(sp.index, name='Property'), sp.reset_index(drop=True).map(lambda v: f(v, 2))], axis=1))
    C = pd.read_csv(os.path.join(T, 'conformal_by_target.csv'))
    C = C[C.cal == 'group'].pivot_table(index='target', columns=['design', 'model'], values='coverage')
    C.index = [LABEL[t] for t in C.index]
    hdr = [f'{d[:4]}–{m}' for d, m in C.columns]
    D = pd.DataFrame([[i] + [f(v, 2) for v in r.values] for i, r in C.iterrows()], columns=['Property'] + hdr)
    save('S7', D)
    L = pd.read_csv(os.path.join(T, 'learning_curves_raw.csv'))
    L['m'] = np.where(L.m_sites >= 11, 'all', L.m_sites.astype(str))
    G = L.groupby(['model', 'm', 'n']).R2.mean().unstack()
    rows = []
    for mod in ['PLSR', 'KRR', 'Cubist']:
        for m in ['2', '3', '4', '6', '8', '10', 'all']:
            if (mod, m) in G.index:
                rows.append([mod, m] + [f(G.loc[(mod, m)].get(n, np.nan)) for n in [100, 200, 400, 800]])
    save('S8', pd.DataFrame(rows, columns=['Model', 'Training sites', 'n = 100', 'n = 200', 'n = 400', 'n = 800']))


if __name__ == '__main__':
    main()
    supp_extra()
