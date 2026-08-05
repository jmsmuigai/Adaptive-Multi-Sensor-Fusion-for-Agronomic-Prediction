"""kenya_figures.py -- all manuscript figures, from real computed outputs only."""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ea as R                     # noqa: E402
from ml import MODELS as ML               # noqa: E402
from cv import metrics                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KEN = os.path.join(HERE, 'ea')
FIG = os.path.join(HERE, 'figures_ea')
os.makedirs(FIG, exist_ok=True)

M = ['PLSR', 'Ridge', 'RandomForest', 'GradientBoosting', 'MLP', 'DualDAFM']
LAB = {'PLSR': 'PLSR', 'Ridge': 'Ridge', 'RandomForest': 'Random forest',
       'GradientBoosting': 'Gradient boosting', 'MLP': 'MLP',
       'DualDAFM': 'Hybrid CNN–DAFM–RBF'}
NICE = {'pH': 'pH', 'OrgC_pct': 'Organic C', 'TotN_pct': 'Total N',
        'P_Olsen': 'Olsen P', 'Ca_exch': 'Exch. Ca', 'Mg_exch': 'Exch. Mg',
        'K_exch': 'Exch. K', 'eCEC': 'eCEC', 'S_total': 'Total S',
        'Fe_total': 'Total Fe', 'Mn_total': 'Total Mn', 'Zn_total': 'Total Zn',
        'Cu_total': 'Total Cu'}
CC = {'Kenya': '#4c72b0', 'Ethiopia': '#dd8452', 'Tanzania': '#55a868',
      'Uganda': '#c44e52'}
DESIGN_LAB = {'cluster': 'Leave-cluster-out', 'site': 'Leave-site-out',
              'country': 'Leave-country-out'}

plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(f, n):
    f.savefig(os.path.join(FIG, n), bbox_inches='tight')
    plt.close(f)
    print('wrote', n)


def fig1():
    """Study area: 16 AfSIS sentinel sites across four East African countries."""
    X, wn, meta, Y = R.load()
    aud = json.load(open(os.path.join(KEN, 'ea_audit.json')))
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9),
                           gridspec_kw={'width_ratios': [1.25, 1, 1.05]})

    for c, g in meta.groupby('Country'):
        ax[0].scatter(g.Longitude, g.Latitude, s=20, alpha=.8, color=CC[c],
                      label=f'{c} (n={len(g)})', edgecolors='k', linewidths=.25)
    ax[0].axhline(0, color='grey', lw=.6, ls=':')
    ax[0].annotate('Equator', (ax[0].get_xlim()[0] + .3, 0.25), fontsize=6.5,
                   color='grey')
    ax[0].set_xlabel('Longitude (\u00b0E)')
    ax[0].set_ylabel('Latitude (\u00b0N)')
    ax[0].set_title(f"(a) {aud['n_sites']} sentinel sites, "
                    f"{len(aud['countries'])} countries")
    ax[0].legend(frameon=False, fontsize=7, loc='best')

    cnt = [aud['per_country'][c]['n'] for c in aud['countries']]
    sts = [aud['per_country'][c]['n_sites'] for c in aud['countries']]
    xs = np.arange(len(aud['countries']))
    ax[1].bar(xs - .2, cnt, .4, color=[CC[c] for c in aud['countries']],
              label='samples')
    ax[1].set_ylabel('Samples')
    a2 = ax[1].twinx()
    a2.bar(xs + .2, sts, .4, color='grey', alpha=.65, label='sites')
    a2.set_ylabel('Sentinel sites')
    a2.grid(False)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(aud['countries'], rotation=20, ha='right')
    ax[1].set_title(f"(b) {aud['n_samples']:,} samples, "
                    f"{aud['n_clusters']} clusters")

    Xs = R.snv(R.savgol(X, 11, 2, 0))
    for c, g in meta.groupby('Country'):
        ax[2].plot(wn, Xs[g.index.values].mean(0), color=CC[c], lw=1.1, label=c)
    ax[2].invert_xaxis()
    ax[2].set_xlabel('Wavenumber (cm$^{-1}$)')
    ax[2].set_ylabel('SNV absorbance')
    ax[2].set_title('(c) Mean MIR spectra by country')
    ax[2].legend(frameon=False, fontsize=7)

    fig.subplots_adjust(wspace=.42)
    fig.suptitle('Figure 1. Study area, sampling structure and mid-infrared spectra',
                 y=1.04)
    save(fig, 'Figure_1_study_area.png')


def fig2():
    """Model comparison under leave-cluster-out validation."""
    r = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    g = r[r.design == 'cluster']
    p = g.pivot(index='target', columns='model', values='R2')[M]
    sd = g.pivot(index='target', columns='model', values='R2_sd')[M]
    order = p.mean(1).sort_values(ascending=False).index
    p, sd = p.loc[order], sd.loc[order]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.1),
                           gridspec_kw={'width_ratios': [1.6, 1]})
    im = ax[0].imshow(p.values, cmap='RdYlGn', vmin=0, vmax=0.9, aspect='auto')
    ax[0].set_xticks(range(len(M)))
    ax[0].set_xticklabels([LAB[m] for m in M], rotation=25, ha='right')
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels([NICE[t] for t in order])
    for i in range(len(order)):
        for j in range(len(M)):
            v = p.values[i, j]
            ax[0].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                       color='black' if 0.15 < v < 0.75 else 'white')
    ax[0].grid(False)
    ax[0].set_title('(a) $R^2$ by model and soil property')
    fig.colorbar(im, ax=ax[0], shrink=.85, label='$R^2$')

    mu = p.mean(0).sort_values()
    cols = ['#c44e52' if k == 'DualDAFM' else '#4c72b0' for k in mu.index]
    ax[1].barh([LAB[k] for k in mu.index], mu.values, color=cols,
               xerr=sd.mean(0)[mu.index].values, capsize=3)
    for i, v in enumerate(mu.values):
        ax[1].text(v + .01, i, f'{v:.3f}', va='center', fontsize=8)
    ax[1].set_xlabel('Mean $R^2$ across 13 soil properties')
    ax[1].set_title('(b) Average performance')
    ax[1].set_xlim(0, mu.max() * 1.25)

    fig.suptitle('Figure 2. Within-landscape prediction (leave-cluster-out cross-validation)',
                 y=1.03)
    save(fig, 'Figure_2_cluster_cv.png')


def fig3():
    """The headline: how performance changes as the validation scale widens."""
    r = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    h = json.load(open(os.path.join(KEN, 'headline.json')))
    designs = ['cluster', 'site', 'country']

    fig, ax = plt.subplots(1, 3, figsize=(12.8, 3.9),
                           gridspec_kw={'width_ratios': [1.15, 1.15, 1]})

    for m in M:
        ys = [h[d]['mean_R2'][m] for d in designs]
        ax[0].plot(range(3), ys, marker='o', lw=1.6 if m == 'DualDAFM' else 1.1,
                   color='#c44e52' if m == 'DualDAFM' else None,
                   zorder=3 if m == 'DualDAFM' else 2, label=LAB[m])
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([DESIGN_LAB[d] for d in designs], rotation=12)
    ax[0].set_ylabel('Mean $R^2$ across 13 properties')
    ax[0].axhline(0, color='k', lw=.8)
    ax[0].set_title('(a) Skill declines as extrapolation widens')
    ax[0].legend(frameon=False, fontsize=7)

    p = r[r.design == 'site'].pivot(index='target', columns='model', values='R2')[M]
    sd = r[r.design == 'site'].pivot(index='target', columns='model',
                                     values='R2_sd')[M]
    o = p.mean(1).sort_values(ascending=False).index
    xs = np.arange(len(o))
    for i, m in enumerate(M):
        ax[1].bar(xs + (i - 2.5) * .14, p.loc[o, m], .14,
                  yerr=sd.loc[o, m], capsize=1.4, error_kw=dict(lw=.6),
                  color='#c44e52' if m == 'DualDAFM' else f'C{i}',
                  label=LAB[m] if False else None)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([NICE[t] for t in o], rotation=40, ha='right', fontsize=7)
    ax[1].axhline(0, color='k', lw=.8)
    ax[1].set_ylim(-1.2, 1)
    ax[1].set_ylabel('$R^2$')
    ax[1].set_title('(b) Leave-site-out, by property')

    npos = {d: [h[d]['n_positive'][m] for m in M] for d in designs}
    w = .26
    for i, d in enumerate(designs):
        ax[2].barh(np.arange(len(M)) + (i - 1) * w, npos[d], w,
                   label=DESIGN_LAB[d])
    ax[2].set_yticks(range(len(M)))
    ax[2].set_yticklabels([LAB[m] for m in M], fontsize=7.5)
    ax[2].set_xlabel('Properties with $R^2 > 0$ (of 13)')
    ax[2].set_title('(c) Coverage by design')
    ax[2].legend(frameon=False, fontsize=7)

    fig.subplots_adjust(wspace=.42)
    fig.suptitle('Figure 3. Predictive skill under three nested spatial '
                 'validation designs', y=1.04)
    save(fig, 'Figure_3_transfer.png')


def fig4():
    """Observed vs predicted, best four properties, leave-cluster-out."""
    X, wn, meta, Y = R.load()
    r = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    g = r[(r.design == 'cluster')]
    best = g.loc[g.groupby('target').R2.idxmax()].nlargest(4, 'R2')
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    for ax, (_, row) in zip(axes, best.iterrows()):
        t, mdl = row.target, row.model
        obs, prd = [], []
        for f in range(1, 5):
            p = os.path.join(KEN, 'parts', f'cluster__{t}__{mdl}__f{f}.json')
            d = json.load(open(p))
            idx = np.array(d['test_idx'])
            obs.append(Y[t].values[idx])
            prd.append(np.array(d['pred']))
        obs, prd = np.concatenate(obs), np.concatenate(prd)
        mm = metrics(obs, prd)
        ax.scatter(obs, prd, s=12, alpha=.45, color='#4c72b0', edgecolors='none')
        lo, hi = np.nanpercentile(np.r_[obs, prd], [1, 99])
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f'Observed {NICE[t]}')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{NICE[t]} ({LAB[mdl]})\n$R^2$={mm["R2"]:.3f}  '
                     f'RMSE={mm["RMSE"]:.3g}', fontsize=8.5)
    fig.suptitle('Figure 4. Out-of-fold observed versus predicted values, '
                 'pooled across leave-cluster-out folds', y=1.06)
    save(fig, 'Figure_4_obs_pred.png')


def fig5():
    """Interpretability: attention over wavenumbers + branch gate."""
    a = pd.read_csv(os.path.join(KEN, 'attention.csv'))
    gt = pd.read_csv(os.path.join(KEN, 'branch_gate.csv'))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6),
                           gridspec_kw={'width_ratios': [1.7, 1]})
    for t, g in a.groupby('target'):
        g = g.sort_values('wavenumber_cm1')
        ax[0].plot(g.wavenumber_cm1, g.attention, lw=1.2, label=NICE[t])
        ax[0].fill_between(g.wavenumber_cm1, g.attention - g.sd,
                           g.attention + g.sd, alpha=.12)
    ax[0].axhline(1.0 / a.position.nunique(), color='k', ls=':', lw=1,
                  label='uniform attention')
    ax[0].invert_xaxis()
    ax[0].set_xlabel('Wavenumber (cm$^{-1}$)')
    ax[0].set_ylabel('Mean DAFM attention')
    ax[0].set_title('(a) Which absorbance regions the model uses')
    ax[0].legend(frameon=False, fontsize=7.5, ncol=2)

    gt = gt.sort_values('spectral_weight')
    ax[1].barh([NICE[t] for t in gt.target], gt.spectral_weight,
               xerr=gt.spectral_sd, capsize=3, color='#55a868', label='spectral')
    ax[1].barh([NICE[t] for t in gt.target], gt.covariate_weight,
               left=gt.spectral_weight, color='#dd8452', label='ancillary')
    ax[1].axvline(.5, color='k', ls=':', lw=1)
    ax[1].set_xlabel('Branch gate weight')
    ax[1].set_title('(b) Measured modality contribution')
    ax[1].legend(frameon=False, fontsize=7.5, loc='lower right')
    ax[1].set_xlim(0, 1)

    fig.subplots_adjust(wspace=.35)
    fig.suptitle('Figure 5. Interpretability of the Dynamic Attention Fusion Module', y=1.04)
    save(fig, 'Figure_5_interpretability.png')


def fig6():
    """Agronomic status vs predictability."""
    d = pd.read_csv(os.path.join(KEN, 'deficiency.csv'))
    r = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    best = r[r.design == 'cluster'].groupby('target').R2.max()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5))
    d = d.sort_values('pct_below')
    ax[0].barh([NICE[t] for t in d.target], d.pct_below, color='#c44e52')
    ax[0].errorbar(d.pct_below, range(len(d)), fmt='none',
                   xerr=[d.pct_below - d.pct_below_half,
                         d.pct_below_1p5 - d.pct_below],
                   ecolor='k', capsize=3, lw=.9)
    ax[0].set_xlabel('% of topsoils below critical level')
    ax[0].set_title('(a) Deficiency prevalence (whiskers: ±50% threshold)')

    ax[1].scatter(d.pct_below, best[d.target].values, s=32, color='#4c72b0')
    for t, x in zip(d.target, d.pct_below):
        ax[1].annotate(NICE[t], (x, best[t]), fontsize=7.5,
                       textcoords='offset points', xytext=(4, 3))
    ax[1].set_xlabel('% deficient')
    ax[1].set_ylabel('Best achievable $R^2$')
    ax[1].set_title('(b) Deficiency versus predictability')
    fig.subplots_adjust(wspace=.32)
    fig.suptitle('Figure 6. Agronomic status of Kenyan topsoils and how well '
                 'each property can be predicted', y=1.04)
    save(fig, 'Figure_6_agronomy.png')




def fig7():
    """Fold-level dispersion: how stable is each model, property by property."""
    fm = pd.read_csv(os.path.join(KEN, 'fold_metrics.csv'))
    c = fm[fm.design == 'cluster']
    order = (c.groupby('target').R2.mean().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.13
    xs = np.arange(len(order))
    for i, m in enumerate(M):
        sub = c[c.model == m]
        for j, t in enumerate(order):
            v = sub[sub.target == t].R2.values
            if not len(v):
                continue
            x = xs[j] + (i - 2.5) * w
            ax.plot([x, x], [v.min(), v.max()], color=f'C{i}', lw=1.1, alpha=.65)
            ax.scatter([x], [v.mean()], s=16, color=f'C{i}', zorder=3,
                       label=LAB[m] if j == 0 else None)
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[t] for t in order], rotation=30, ha='right')
    ax.set_ylabel('$R^2$ per fold')
    ax.set_ylim(-0.6, 1.0)
    ax.legend(frameon=False, fontsize=7.5, ncol=6, loc='lower left')
    ax.set_title('Figure 7. Fold-level spread of $R^2$ under leave-cluster-out '
                 'cross-validation. Vertical bars span the four evaluation folds; '
                 'markers are fold means.', fontsize=9.5)
    save(fig, 'Figure_7_fold_spread.png')


def fig8():
    """Residual diagnostics for the best-predicted properties."""
    X, wn, meta, Y = R.load()
    r = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    g = r[r.design == 'cluster']
    best = g.loc[g.groupby('target').R2.idxmax()].nlargest(3, 'R2')
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for k, (_, row) in enumerate(best.iterrows()):
        t, mdl = row.target, row.model
        obs, prd, site = [], [], []
        for f in range(1, 5):
            d = json.load(open(os.path.join(
                KEN, 'parts', f'cluster__{t}__{mdl}__f{f}.json')))
            idx = np.array(d['test_idx'])
            obs.append(Y[t].values[idx]); prd.append(np.array(d['pred']))
            site.append(meta.Country.values[idx])
        obs = np.concatenate(obs); prd = np.concatenate(prd)
        site = np.concatenate(site); res = prd - obs

        a = axes[0, k]
        a.scatter(prd, res, s=11, alpha=.45, color='#4c72b0', edgecolors='none')
        a.axhline(0, color='r', ls='--', lw=1)
        a.set_xlabel(f'Predicted {NICE[t]}'); a.set_ylabel('Residual')
        a.set_title(f'{NICE[t]} — residuals vs fitted', fontsize=9)

        b = axes[1, k]
        for s in CC:
            m_ = site == s
            if m_.any():
                b.hist(res[m_], bins=18, alpha=.55, color=CC[s], label=s)
        b.axvline(0, color='k', lw=.8)
        b.set_xlabel('Residual'); b.set_ylabel('Count')
        b.set_title(f'{NICE[t]} — residuals by country', fontsize=9)
        b.legend(frameon=False, fontsize=7)
    fig.suptitle('Figure 8. Residual diagnostics for the three best-predicted '
                 'properties, pooled across leave-cluster-out folds', y=1.02)
    fig.subplots_adjust(hspace=.42, wspace=.32)
    save(fig, 'Figure_8_residuals.png')


def fig9():
    """Spectral detail: absorbance regions and where attention concentrates."""
    X, wn, meta, Y = R.load()
    a = pd.read_csv(os.path.join(KEN, 'attention.csv'))
    Xs = R.snv(R.savgol(X, 11, 2, 0))
    d1 = R.snv(R.savgol(X, 11, 2, 1))
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True)
    ax[0].plot(wn, Xs.mean(0), color='#333', lw=1.2, label='mean SNV absorbance')
    ax[0].fill_between(wn, Xs.mean(0) - Xs.std(0), Xs.mean(0) + Xs.std(0),
                       color='#999', alpha=.3, label='±1 SD across samples')
    bands = [(3600, 3700, 'O–H'), (2850, 2960, 'C–H'), (1600, 1650, 'C=O / amide'),
             (1400, 1480, 'carbonate'), (900, 1100, 'Si–O')]
    for lo, hi, nm in bands:
        ax[0].axvspan(lo, hi, color='#4c72b0', alpha=.10)
        ax[0].annotate(nm, ((lo + hi) / 2, ax[0].get_ylim()[1] * .82),
                       ha='center', fontsize=7, color='#26456e')
    ax[0].set_ylabel('SNV absorbance')
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title('(a) Mean mid-infrared spectrum with principal absorbance regions')

    for t, gg in a.groupby('target'):
        gg = gg.sort_values('wavenumber_cm1')
        ax[1].plot(gg.wavenumber_cm1, gg.attention, lw=1.2, label=NICE[t])
    for lo, hi, _ in bands:
        ax[1].axvspan(lo, hi, color='#4c72b0', alpha=.10)
    ax[1].axhline(1.0 / a.position.nunique(), color='k', ls=':', lw=1)
    ax[1].set_xlabel('Wavenumber (cm$^{-1}$)')
    ax[1].set_ylabel('Mean DAFM attention')
    ax[1].legend(frameon=False, fontsize=7.5, ncol=4)
    ax[1].set_title('(b) Attention aligned to the same wavenumber axis')
    ax[1].invert_xaxis()
    fig.suptitle('Figure 9. Attention placed in spectroscopic context', y=1.01)
    save(fig, 'Figure_9_spectral_context.png')






def fig10():
    """Per-country transfer performance."""
    det = pd.read_csv(os.path.join(KEN, 'country_detail.csv'))
    r = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7),
                           gridspec_kw={'width_ratios': [1.1, 1.25, 1]})

    piv = det.pivot_table(index='country', columns='model', values='R2')[M]
    xs = np.arange(len(piv.index))
    for i, m in enumerate(M):
        ax[0].bar(xs + (i - 2.5) * .14, piv[m], .14,
                  color='#c44e52' if m == 'DualDAFM' else f'C{i}',
                  label=LAB[m])
    ax[0].set_xticks(xs); ax[0].set_xticklabels(piv.index)
    ax[0].axhline(0, color='k', lw=.8)
    ax[0].set_ylabel('Mean $R^2$ over 13 properties')
    ax[0].set_title('(a) Skill when each country is withheld')
    ax[0].legend(frameon=False, fontsize=6.5, ncol=2)

    bp = det.pivot_table(index='target', columns='country', values='R2', aggfunc='max')
    o = bp.mean(1).sort_values(ascending=False).index
    im = ax[1].imshow(bp.loc[o].values, cmap='RdYlGn', vmin=-0.5, vmax=0.9,
                      aspect='auto')
    ax[1].set_xticks(range(bp.shape[1])); ax[1].set_xticklabels(bp.columns)
    ax[1].set_yticks(range(len(o)))
    ax[1].set_yticklabels([NICE[t] for t in o], fontsize=7)
    for i in range(len(o)):
        for j in range(bp.shape[1]):
            ax[1].text(j, i, f'{bp.loc[o[i]].values[j]:.2f}', ha='center',
                       va='center', fontsize=6.5)
    ax[1].grid(False)
    ax[1].set_title('(b) Best $R^2$ per property and held-out country')
    fig.colorbar(im, ax=ax[1], shrink=.85)

    cnt = det.groupby('country').apply(
        lambda g: (g.groupby('model').R2.mean().max()))
    npos = det.groupby('country').apply(
        lambda g: int((g[g.model == g.groupby('model').R2.mean().idxmax()].R2 > 0).sum()))
    ax[2].scatter(cnt.values, npos.values, s=60, color='#4c72b0')
    for c in cnt.index:
        ax[2].annotate(c, (cnt[c], npos[c]), textcoords='offset points',
                       xytext=(6, 4), fontsize=8)
    ax[2].set_xlabel('Best mean $R^2$')
    ax[2].set_ylabel('Properties with $R^2 > 0$')
    ax[2].set_title('(c) Transfer difficulty by country')

    fig.subplots_adjust(wspace=.42)
    fig.suptitle('Figure 10. Leave-country-out transfer, resolved by held-out country',
                 y=1.04)
    save(fig, 'Figure_10_country.png')


ALL = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5,
       'fig6': fig6, 'fig7': fig7, 'fig8': fig8, 'fig9': fig9, 'fig10': fig10}

if __name__ == '__main__':
    for w in (sys.argv[1:] or list(ALL)):
        ALL[w]()
