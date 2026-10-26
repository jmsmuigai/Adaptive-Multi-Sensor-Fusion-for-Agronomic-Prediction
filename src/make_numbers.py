"""numbers.py -- every number quoted in the manuscript text, from out/tables."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, LABEL, OUT, SITE_AEZ
T = os.path.join(OUT, 'tables')


def main():
    X, wnb, meta, Y = load()
    N = {}
    # ----------------------------------------------------------- data
    N['n_tiny'] = int(meta.SiteKey.isin(['Tanzania_Bukwaya', 'Tanzania_Kiberashi', 'Tanzania_Mbinga']).sum())
    for z in ['Aw', 'Am', 'BSh', 'BWh', 'C']:
        N[f'aez_n_{z}'] = int((meta.AEZ == z).sum())
    p = Y.P_Olsen.dropna()
    N['p_median'] = round(float(p.median()), 1)
    N['p_below2'] = float((p < 2).mean() * 100)
    N['n_bins'] = int(X.shape[1])
    binw = float(np.median(np.diff(wnb)))
    N['bin_width'] = binw
    parts = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'parts', 'site__*__Hybrid__f*.json'))]
    rfs = [(s['rf_bins'], s['jump_bins']) for r in parts if not r.get('variant') for s in r['seeds'][:1]]
    rf, jp = pd.Series([a for a, b in rfs]).mode()[0], pd.Series([b for a, b in rfs]).mode()[0]
    N['rf_bins'] = int(rf); N['rf_cm'] = rf * binw; N['jump_cm'] = jp * binw
    N['rf_max_cm'] = max(a for a, b in rfs) * binw
    import sklearn, lightgbm
    N['sk_ver'] = sklearn.__version__; N['lgb_ver'] = lightgbm.__version__
    # fits
    allp = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'parts', '*.json'))]
    nf = 0
    for r in allp:
        if 'seeds' in r:
            nf += len(r['seeds']) + len(r.get('tune_val', []))
        else:
            nf += 1 + len(r.get('inner_scores', [])) * (3 if r['model'] in ('PLSR', 'Ridge', 'SVR', 'KRR') else 1)
    for name in ['exp_spike', 'exp_lc', 'exp_kenya', 'exp_conformal', 'exp_prep', 'agreement']:
        for base in [os.path.join(OUT, 'vm', 'exp'), os.path.join(OUT, 'exp')]:
            for f in glob.glob(os.path.join(base, name, '*.json')):
                r = json.load(open(f)); nf += len(r) if isinstance(r, list) else 8
    N['n_fits_total'] = int(round(nf, -2))
    # agreement
    ag = pd.read_csv(os.path.join(OUT, 'agreement.csv'))
    md = ag.groupby('model').apply(lambda g: np.median(np.abs(g.R2 - g.R2_lib)))
    N['agree_rf'] = md['RandomForest']; N['agree_gbm'] = md['GradientBoosting']; N['agree_pls'] = md['PLSR']
    N['agree_ridge'] = md['Ridge']
    # ----------------------------------------------------------- main
    S = pd.read_csv(os.path.join(T, 'pooled_metrics_all.csv'))
    M = S.groupby(['design', 'model']).R2.mean().unstack(0)
    for d in M.columns:
        b = M[d].idxmax()
        N[f'best_{d}'] = b; N[f'best_{d}_r2'] = M[d].max()
        N[f'hyb_{d}_r2'] = M.loc['Hybrid', d]
        N[f'plsr_{d}_r2'] = M.loc['PLSR', d]
        N[f'range_{d}_lo'] = M[d].min(); N[f'range_{d}_hi'] = M[d].max()
        N[f'spread_{d}'] = M[d].max() - M[d].min()
        N[f'spread_nohyb_{d}'] = M[d].drop('Hybrid').max() - M[d].drop('Hybrid').min()
        N[f'median_model_{d}'] = M[d].median()
    for d in M.columns:
        N[f'drop_{d}_pct'] = 100 * (1 - M[d].median() / M['random'].median())
    F = pd.read_csv(os.path.join(T, 'friedman.csv'))
    for d, g in F.groupby('design'):
        N[f'fried_p_{d}'] = g.p.iloc[0]; N[f'fried_chi_{d}'] = g.chi2.iloc[0]; N[f'cd_{d}'] = g.cd.iloc[0]
        g = g.sort_values('mean_rank')
        N[f'rank1_{d}'] = g.model.iloc[0]; N[f'rank1_{d}_v'] = g.mean_rank.iloc[0]
        N[f'rankhyb_{d}'] = g.set_index('model').mean_rank['Hybrid']
        N[f'n_within_cd_{d}'] = int((g.mean_rank <= g.mean_rank.iloc[0] + g.cd.iloc[0]).sum())
    PR = pd.read_csv(os.path.join(T, 'boot_pairs.csv'))
    # GPR vs others on site: how many pairwise differences significant
    for d in ['site', 'country', 'aez']:
        best = N[f'best_{d}']
        sub = PR[(PR.design == d) & ((PR.a == best) | (PR.b == best))].copy()
        sub['other'] = np.where(sub.a == best, sub.b, sub.a)
        sub['dd'] = np.where(sub.a == best, sub.d_mean, -sub.d_mean)
        sig = sub[(sub.p < 0.05) & (sub.dd > 0)]
        N[f'best_{d}_sig_wins'] = len(sig); N[f'best_{d}_pairs'] = len(sub)
        hy = PR[(PR.design == d) & ((PR.a == 'Hybrid') | (PR.b == 'Hybrid'))].copy()
        hy['other'] = np.where(hy.a == 'Hybrid', hy.b, hy.a)
        hy['dd'] = np.where(hy.a == 'Hybrid', hy.d_mean, -hy.d_mean)
        N[f'hyb_{d}_sig_losses'] = int(((hy.p < 0.05) & (hy.dd < 0)).sum())
        N[f'hyb_{d}_sig_wins'] = int(((hy.p < 0.05) & (hy.dd > 0)).sum())
        N[f'hyb_{d}_n_pairs'] = len(hy)
        # hybrid vs CNN on site, per property
        hc = hy[hy.other == 'CNN']
        N[f'hyb_cnn_{d}_sig'] = int(((hc.p < 0.05) & (hc.dd < 0)).sum())
    site = S[S.design == 'site']
    bestprop = site.groupby('target').R2.max()
    for t in TARGETS:
        N[f'site_best_{t}'] = bestprop[t]
        r = site[site.target == t].set_index('model')
        N[f'site_hyb_{t}'] = r.R2['Hybrid']
        N[f'site_gpr_{t}'] = r.R2['GPR']; N[f'site_gpr_rpd_{t}'] = r.RPD['GPR']
        N[f'site_bestmodel_{t}'] = r.R2.idxmax()
        N[f'site_rpd_{t}'] = r.RPD.max(); N[f'site_rpiq_{t}'] = r.RPIQ.max(); N[f'site_ccc_{t}'] = r.CCC.max()
        rr = S[(S.design == 'random') & (S.target == t)].R2.max()
        N[f'rand_best_{t}'] = rr
        N[f'country_best_{t}'] = S[(S.design == 'country') & (S.target == t)].R2.max()
    N['n_rpd2_gpr'] = int((site[site.model == 'GPR'].RPD >= 2).sum())
    for v in ['pool_mean', 'no_pos', 'spectra_only']:
        pass
    N['n_rpd2_site'] = int((site.groupby('target').RPD.max() >= 2).sum())
    N['n_r2gt0_country_best'] = int((S[S.design == 'country'].groupby('target').R2.max() > 0).sum())
    # unclipped failures
    un = S.groupby(['design', 'model']).R2_unclipped.min().unstack(0)
    N['worst_unclipped'] = float(un.min().min())
    w = S.loc[S.R2_unclipped.idxmin()]
    N['worst_unclipped_desc'] = f"{w.model}, {LABEL[w.target]}, {w.design}"
    N['n_unclipped_lt_m1'] = int((S.R2_unclipped < -1).sum())
    N['n_unclipped_lt_m1_hyb'] = int(((S.R2_unclipped < -1) & (S.model == 'Hybrid')).sum())
    bad = S[S.R2_unclipped < -1]
    N['unclipped_fail_models'] = ', '.join(sorted(bad.model.unique()))
    N['unclipped_fail_targets'] = ', '.join(sorted({LABEL[t] for t in bad.target}))
    N['seed_sd_hyb_site'] = S[(S.design == 'site') & (S.model == 'Hybrid')].R2_seed_sd.mean()
    N['seed_sd_cnn_site'] = S[(S.design == 'site') & (S.model == 'CNN')].R2_seed_sd.mean()
    N['seed_sd_mlp_site'] = S[(S.design == 'site') & (S.model == 'MLP')].R2_seed_sd.mean()
    # ----------------------------------------------------------- ablation
    D = pd.read_csv(os.path.join(T, 'ablation.csv'))
    Dc = D[D.clipped]
    for v, g in Dc[Dc.design == 'site'].groupby('variant'):
        N[f'abl_{v}'] = g.dR2.mean(); N[f'abl_{v}_nb'] = int((g.dR2 > 0).sum())
        N[f'abl_{v}_nsig'] = int(((g.d_lo > 0) | (g.d_hi < 0)).sum())
        N[f'abl_{v}_nsigpos'] = int((g.d_lo > 0).sum()); N[f'abl_{v}_nsigneg'] = int((g.d_hi < 0).sum())
    B = pd.read_csv(os.path.join(T, 'boundedness.csv'))
    for _, r in B.iterrows():
        k = f"{r.design}_{'c' if r.clipped else 'u'}"
        for c in ['worst_rbf', 'worst_lin', 'worstlog_rbf', 'worstlog_lin', 'p10_rbf', 'p10_lin', 'mean_rbf', 'mean_lin']:
            N[f'bd_{k}_{c}'] = r[c]
        N[f'bd_{k}_nlin'] = int(r.n_lin_better)
    misc = json.load(open(os.path.join(T, 'misc.json')))
    N.update({k: v for k, v in misc['bound'].items()})
    for k, v in misc['p_ceiling'].items():
        N[f'pceil_{k}_orig'] = v['orig']; N[f'pceil_{k}_log'] = v['log']
    # ----------------------------------------------------------- kenya / lc / spike / conformal
    K = pd.read_csv(os.path.join(T, 'kenya.csv'))
    K = K[K.model.isin(['PLSR', 'KRR', 'Cubist', 'Hybrid'])]
    Km = K.groupby(['model', 'cond']).R2.mean().unstack()
    for m in Km.index:
        for c in Km.columns:
            N[f'ken_{m}_{c.replace("+", "p")}'] = Km.loc[m, c]
    Ka = K.groupby('cond').R2.mean()
    for c in Ka.index:
        N[f'ken_all_{c.replace("+", "p")}'] = Ka[c]
    N['ken_n_K2'] = K[K.cond == 'K2'].n_train.mean(); N['ken_n_K2pEA'] = K[K.cond == 'K2+EA'].n_train.mean()
    N['ken_sites_K2pEA_n'] = K[K.cond == 'K2+EA_n'].n_sites.mean()
    Kt = K.groupby(['target', 'cond']).R2.mean().unstack()
    N['ken_n_props_div_beats_k2'] = int((Kt['K2+EA_n'] > Kt['K2']).sum())
    N['ken_n_props_full_beats_ea'] = int((Kt['K2+EA'] > Kt['EA']).sum())
    L = pd.read_csv(os.path.join(T, 'learning_curves_raw.csv'))
    L['m'] = np.where(L.m_sites >= 11, 'all', L.m_sites.astype(str))
    G = L.groupby(['model', 'm', 'n']).R2.mean()
    for mod in ['KRR', 'PLSR', 'Cubist']:
        for m in ['2', '4', '6', '8', 'all']:
            for n in [100, 200, 400, 800]:
                if (mod, m, n) in G.index:
                    N[f'lc_{mod}_{m}_{n}'] = G[(mod, m, n)]
    Sp = pd.read_csv(os.path.join(T, 'spiking.csv'))
    Sm = Sp.groupby(['model', 'k']).R2.mean()
    for (m, k), v in Sm.items():
        N[f'sp_{m}_{k}'] = v
    for t in ['P_Olsen', 'K_exch', 'pH', 'OrgC_pct']:
        for m in ['KRR', 'PLSR', 'LightGBM', 'Hybrid']:
            for k in [0, 50]:
                r = Sp[(Sp.target == t) & (Sp.model == m) & (Sp.k == k)]
                if len(r):
                    N[f'sp_{m}_{t}_{k}'] = r.R2.iloc[0]
    Cf = pd.read_csv(os.path.join(T, 'conformal.csv'))
    for _, r in Cf.iterrows():
        N[f'cf_{r.design}_{r.model}_{r.cal}'] = r.coverage
    for d in ['site', 'aez', 'country']:
        for c in ['group', 'random']:
            N[f'cf_{d}_{c}_mean'] = Cf[(Cf.design == d) & (Cf.cal == c)].coverage.mean()
    Pp = pd.read_csv(os.path.join(T, 'prep.csv'), index_col=0)
    for v in Pp.index:
        key = {'SG1+SNV, CO2 masked (adopted)': 'adopted', 'SG1+SNV, unmasked (round 1)': 'unmasked',
               'SNV then SG1, masked': 'snvsg', 'SNV only, masked': 'snvonly', 'SG2+SNV, masked': 'sg2',
               'SG1+SNV, masked, bin x8': 'bin8'}[v]
        N[f'prep_{key}_KRR'] = Pp.loc[v, 'KRR']; N[f'prep_{key}_PLSR'] = Pp.loc[v, 'PLSR']
    # ----------------------------------------------------------- interp
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'exp', 'exp_interp', '*.json'))]
    sp = pd.DataFrame([r['spearman'] for r in rows])
    for c in sp.columns:
        N['rho_' + c.replace('~', '_')] = sp[c].mean()
    ov = pd.DataFrame([dict(target=r['target'], method=k, obs=v[0], null=v[1], p=v[2])
                       for r in rows if 'overlap' in r for k, v in r['overlap'].items()])
    for (t, m), g in ov.groupby(['target', 'method']):
        chi = -2 * np.log(g.p).sum()
        N[f'ov_{t}_{m}_ratio'] = g.obs.mean() / g.null.mean()
        N[f'ov_{t}_{m}_fisher'] = float(stats.chi2.sf(chi, 2 * len(g)))
        N[f'ov_{t}_{m}_nsig'] = int((g.p < 0.05).sum())
    pm = pd.DataFrame([dict(target=r['target'], d=r['perm_depth_dMSE'], s=r['perm_spec_dMSE'], g=r['gate_spec'])
                       for r in rows]).groupby('target').mean()
    for t in pm.index:
        N[f'perm_depth_{t}'] = pm.loc[t, 'd']; N[f'perm_spec_{t}'] = pm.loc[t, 's']; N[f'gate_{t}'] = pm.loc[t, 'g']
    N['gate_min'] = pm.g.min(); N['gate_max'] = pm.g.max()
    N['gate_rho_spec'] = float(stats.spearmanr(pm.g, pm.s / (pm.s + pm.d)).correlation)
    # ----------------------------------------------------------- screening
    sc = pd.read_csv(os.path.join(T, 'screening.csv'))
    best = N['best_site']
    for _, r in sc[sc.model == best].iterrows():
        for c in ['sens', 'spec', 'bal_acc', 'prevalence']:
            N[f'scr_{r.target}_{c}'] = r[c]
    import analyze as A
    pl = A.load_parts('site__*__LightGBM__*')
    a = A.assemble(pl, 'site', 'LightGBM', 'plus_coords'); b = A.assemble(pl, 'site', 'LightGBM', '')
    dd = []
    for t in TARGETS:
        ia, Pa, _ = a[t]; ib, Pb, _ = b[t]; oa, ob = np.argsort(ia), np.argsort(ib); y = Y[t].values[ia[oa]]
        r = lambda P: 1 - ((P - y) ** 2).sum(1) / ((y - y.mean()) ** 2).sum()
        dd.append(r(Pa[:, oa]).mean() - r(Pb[:, ob]).mean())
    N['lgb_coords_d'] = float(np.mean(dd)); N['lgb_coords_nb'] = int(sum(x > 0 for x in dd))
    N['n_within_cd_site_m1'] = N['n_within_cd_site'] - 1
    N['ken_breadth_effect'] = N['ken_all_K2pEA'] - N['ken_all_K2']
    N['plsr_rank_site'] = int(S[S.design == 'site'].groupby('model').R2.mean().rank(ascending=False)['PLSR'])
    for v in ['pool_mean', 'no_pos', 'spectra_only']:
        N[f'abl_{v}_abs'] = abs(N[f'abl_{v}'])
    json.dump(N, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)
    return N


if __name__ == '__main__':
    N = main()
    for k in sorted(N):
        print(k, N[k])
