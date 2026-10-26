"""Figures 3-10 of the revised manuscript, built from out/tables (results.py)."""
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, LABEL, OUT
import analyze as A

T = os.path.join(OUT, 'tables'); F = os.path.join(OUT, 'fig'); os.makedirs(F, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                                                           '#e87ba4', '#008300', '#6250d6', '#e34948')
GREY, INK, INK2 = '#b9b8b2', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'savefig.dpi': 300})
HL = {'Hybrid': ORANGE, 'GPR': BLUE, 'CNN': AQUA, 'PLSR': VIOLET}
DES = ['random', 'cluster', 'site', 'aez', 'country']
DLAB = ['Random', 'Cluster', 'Site', 'AEZ', 'Country']
SHORT = {t: LABEL[t] for t in TARGETS}


def tag(ax, s):
    ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')


def save(name):
    plt.savefig(os.path.join(F, name), bbox_inches='tight', facecolor='white')
    plt.close()


S = pd.read_csv(os.path.join(T, 'pooled_metrics_all.csv'))
CI = pd.read_csv(os.path.join(T, 'boot_ci.csv'))

# ---------------------------------------------------------------- Fig 3
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={'width_ratios': [1.25, 1]})
M = S.groupby(['model', 'design']).R2.mean().unstack()
ax = axs[0]
for m in M.index:
    v = [M.loc[m].get(d, np.nan) for d in DES]
    if m in HL:
        continue
    ax.plot(range(5), v, color=GREY, lw=1.0, marker='o', ms=3, zorder=1)
for m, c in HL.items():
    if m in M.index:
        v = [M.loc[m].get(d, np.nan) for d in DES]
        ax.plot(range(5), v, color=c, lw=2, marker='o', ms=5, zorder=3, label=m)
ax.plot([], [], color=GREY, lw=1, marker='o', ms=3, label='other 9 models')
ax.set_xticks(range(5)); ax.set_xticklabels(DLAB)
ax.set_ylabel('Mean pooled R$^2$ (13 properties)'); ax.set_xlabel('Withheld unit')
ax.legend(loc='lower left', fontsize=7.5); tag(ax, '(a)')
ax = axs[1]
best = S[S.design.isin(['random', 'site', 'country'])].groupby(['design', 'target']).R2.max().unstack(0)
best = best.loc[best['random'].sort_values().index]
y = np.arange(len(best))
for d, c, mk in [('random', GREY, 'o'), ('site', BLUE, 's'), ('country', ORANGE, 'D')]:
    ax.scatter(best[d], y, color=c, s=22, marker=mk, zorder=3, label={'random': 'Random', 'site': 'Leave-site-out', 'country': 'Leave-country-out'}[d],
               edgecolor='white', linewidth=0.6)
for i, t in enumerate(best.index):
    ax.plot([best.loc[t].min(), best.loc[t].max()], [i, i], color='#d4d3cd', lw=1.2, zorder=1)
ax.set_yticks(y); ax.set_yticklabels([SHORT[t] for t in best.index])
ax.set_xlabel('Best-model pooled R$^2$'); ax.legend(loc='lower right', fontsize=7)
ax.set_xlim(-0.05, 1.0); tag(ax, '(b)')
plt.tight_layout(); save('Figure_3_designs.png')

# ---------------------------------------------------------------- Fig 4
site = S[S.design == 'site']
order = site.groupby('target').R2.max().sort_values().index
fig, ax = plt.subplots(figsize=(7.6, 4.6))
y = np.arange(len(order))
others = [m for m in A.MODELS_ORDER if m not in HL]
for i, t in enumerate(order):
    v = site[(site.target == t) & site.model.isin(others)].R2
    ax.scatter(v, np.full(len(v), i), color=GREY, s=10, zorder=2)
off = {'Hybrid': 0.22, 'GPR': -0.22, 'CNN': 0.0, 'PLSR': 0.11}
for m, c in HL.items():
    for i, t in enumerate(order):
        r = site[(site.target == t) & (site.model == m)]
        ci = CI[(CI.design == 'site') & (CI.target == t) & (CI.model == m)]
        if r.empty:
            continue
        if not ci.empty:
            ax.plot([ci.R2_lo.iloc[0], ci.R2_hi.iloc[0]], [i + off[m]] * 2, color=c, lw=1.4, zorder=3)
        ax.scatter(r.R2, [i + off[m]], color=c, s=26, zorder=4, edgecolor='white', linewidth=0.6,
                   label=m if i == 0 else None)
ax.scatter([], [], color=GREY, s=10, label='other 9 models')
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([SHORT[t] for t in order])
ax.set_xlabel('Pooled out-of-fold R$^2$, leave-site-out (bars: 95% group-bootstrap interval)')
ax.legend(loc='lower right', fontsize=7.5)
plt.tight_layout(); save('Figure_4_site_properties.png')

# ---------------------------------------------------------------- Fig 5 obs vs pred
preds = pickle.load(open(os.path.join(OUT, 'preds.pkl'), 'rb'))
X, wnb, meta, Y = load()
fig, axs = plt.subplots(2, 4, figsize=(9.6, 4.9))
for j, t in enumerate(['OrgC_pct', 'pH', 'eCEC', 'P_Olsen']):
    for i, m in enumerate(['GPR', 'Hybrid']):
        ax = axs[i, j]
        if ('site', m, t) not in preds:
            continue
        idx, P = preds[('site', m, t)]
        y = Y[t].values[idx]; p = P.mean(0)
        ax.scatter(y, p, s=5, alpha=0.45, color=HL[m], edgecolor='none')
        lo, hi = np.percentile(np.r_[y, p], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls='--')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if t in ('P_Olsen',):
            ax.set_xscale('symlog', linthresh=2); ax.set_yscale('symlog', linthresh=2)
        r2 = S[(S.design == 'site') & (S.model == m) & (S.target == t)].R2.iloc[0]
        ax.set_title(f'{SHORT[t]} - {m}  (R$^2$ = {r2:.2f})', fontsize=8)
        if j == 0:
            ax.set_ylabel('Predicted')
        if i == 1:
            ax.set_xlabel('Observed')
plt.tight_layout(); save('Figure_5_obs_pred.png')

# ---------------------------------------------------------------- Fig 6 ablation
D = pd.read_csv(os.path.join(T, 'ablation.csv'))
D = D[(D.design == 'site') & (D.clipped)]
lab = {'head_linear': 'Linear head', 'head_mlp': 'Dense (ReLU) head', 'head_gauss': 'Round-1 Gaussian head',
       'pool_mean': 'Mean pooling', 'pool_reweight': 'Attention re-weighting', 'no_pos': 'No positional embedding',
       'fuse_concat': 'Concatenation, no gate', 'spectra_only': 'Spectra only (no depth)',
       'plus_coords': '+ latitude/longitude'}
agg = D.groupby('variant').agg(d=('dR2', 'mean'), lo=('dR2', lambda x: np.percentile(x, 10)),
                               hi=('dR2', lambda x: np.percentile(x, 90)), nb=('dR2', lambda x: int((x > 0).sum())))
agg = agg.loc[[v for v in lab if v in agg.index]].sort_values('d')
fig, ax = plt.subplots(figsize=(7.2, 3.6))
yy = np.arange(len(agg))
for i, (v, r) in enumerate(agg.iterrows()):
    pts = D[D.variant == v].dR2.values
    ax.scatter(pts, np.full(len(pts), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(pts)),
               s=8, color=GREY, zorder=2)
    c = BLUE if r.d > 0 else ORANGE
    ax.barh(i, r.d, height=0.5, color=c, alpha=0.85, zorder=3)
    ax.text(1.01, i, f'{r.d:+.3f}   {int(r.nb)}/13', va='center', ha='left', fontsize=7.4, color=INK,
            transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
ax.text(1.01, len(agg) - 0.4, 'mean dR$^2$   better', transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
        fontsize=7, color=INK2, ha='left')
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([lab[v] for v in agg.index])
ax.set_xlabel('Change in pooled R$^2$ relative to the full hybrid (leave-site-out); dots = properties')
plt.tight_layout(); save('Figure_6_ablation.png')

# ---------------------------------------------------------------- Fig 7 size vs diversity
L = pd.read_csv(os.path.join(T, 'learning_curves_raw.csv'))
L['m'] = np.where(L.m_sites >= 11, 'all', L.m_sites.astype(str))
K = pd.read_csv(os.path.join(T, 'kenya.csv'))
fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={'width_ratios': [1, 1, 1.05]})
for ax, mod, tg in [(axs[0], 'KRR', '(a)'), (axs[1], 'PLSR', '(b)')]:
    G = L[L.model == mod].groupby(['m', 'n']).R2.mean().unstack()
    ms = [m for m in ['2', '3', '4', '6', '8', '10', 'all'] if m in G.index]
    for n, c in zip([100, 200, 400, 800], ['#c7dcf3', '#7fb0e6', BLUE, '#123f7a']):
        v = [G.loc[m].get(n, np.nan) for m in ms]
        ax.plot(range(len(ms)), v, marker='o', ms=4, lw=1.8, color=c, label=f'n = {n}')
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms)
    ax.set_xlabel('Number of training sites'); ax.set_ylabel('Mean pooled R$^2$ (13 properties)')
    ax.set_title(mod, fontsize=9); tag(ax, tg)
    if mod == 'KRR':
        ax.legend(fontsize=7, loc='lower right')
ax = axs[2]
conds = ['K2', 'K2+EA_n', 'EA', 'K2+EA']
cl = {'K2': 'Kenya, 2 sites', 'K2+EA_n': 'Diverse,\nsize-matched', 'EA': 'Other\ncountries', 'K2+EA': 'Kenya +\nother'}
Kk = K[K.model.isin(['PLSR', 'KRR', 'Cubist', 'Hybrid'])].groupby(['model', 'cond']).R2.mean().unstack()
w = 0.2
for i, (m, c) in enumerate(zip(['PLSR', 'KRR', 'Cubist', 'Hybrid'], [VIOLET, BLUE, AQUA, ORANGE])):
    if m in Kk.index:
        ax.bar(np.arange(4) + (i - 1.5) * w, [Kk.loc[m].get(cd, np.nan) for cd in conds], width=w * 0.92,
               color=c, label=m)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([cl[c] for c in conds], fontsize=7.2)
ax.set_ylabel('Mean pooled R$^2$ on the three Kenyan sites'); ax.legend(fontsize=7, ncol=2, loc='upper left')
tag(ax, '(c)')
plt.tight_layout(); save('Figure_7_size_diversity.png')

# ---------------------------------------------------------------- Fig 8 spiking
Sp = pd.read_csv(os.path.join(T, 'spiking.csv'))
fig, axs = plt.subplots(1, 4, figsize=(10, 2.9), sharex=True)
for ax, t, tg in zip(axs, ['MEAN', 'pH', 'OrgC_pct', 'P_Olsen'], ['(a)', '(b)', '(c)', '(d)']):
    for m, c in [('PLSR', VIOLET), ('KRR', BLUE), ('LightGBM', AQUA), ('Hybrid', ORANGE)]:
        g = Sp[Sp.model == m]
        g = g.groupby('k').R2.mean() if t == 'MEAN' else g[g.target == t].set_index('k').R2
        ax.plot(g.index, g.values, marker='o', ms=4, lw=1.8, color=c, label=m)
    ax.set_title('Mean of 13 properties' if t == 'MEAN' else SHORT[t], fontsize=9)
    ax.set_xlabel('Local samples added (k)'); ax.set_xticks([0, 5, 10, 20, 50]); tag(ax, tg)
axs[0].set_ylabel('Pooled R$^2$, leave-site-out'); axs[0].legend(fontsize=7)
plt.tight_layout(); save('Figure_8_spiking.png')

# ---------------------------------------------------------------- Fig 9 conformal
C = pd.read_csv(os.path.join(T, 'conformal_by_target.csv'))
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
for ax, d, tg in zip(axs, ['site', 'aez', 'country'], ['(a)', '(b)', '(c)']):
    for i, (m, c) in enumerate([('KRR', BLUE), ('LightGBM', AQUA), ('Hybrid', ORANGE)]):
        for j, cal in enumerate(['random', 'group']):
            v = C[(C.design == d) & (C.model == m) & (C.cal == cal)].coverage.values
            x = i * 2.6 + j
            ax.scatter(np.full(len(v), x) + np.random.default_rng(i * 10 + j).uniform(-0.15, 0.15, len(v)), v,
                       s=9, color=c if cal == 'group' else GREY, zorder=2)
            ax.plot([x - 0.3, x + 0.3], [v.mean()] * 2, color=INK, lw=1.6, zorder=3)
    ax.axhline(0.9, color=RED, lw=1, ls='--')
    ax.set_xticks([0, 1, 2.6, 3.6, 5.2, 6.2])
    ax.set_xticklabels(['KRR\nrand.', 'KRR\ngroup', 'LGBM\nrand.', 'LGBM\ngroup', 'Hybrid\nrand.', 'Hybrid\ngroup'], fontsize=6.8)
    ax.set_title({'site': 'Leave-site-out', 'aez': 'Leave-AEZ-out', 'country': 'Leave-country-out'}[d], fontsize=9)
    tag(ax, tg)
axs[0].set_ylabel('Empirical coverage of 90% intervals\n(dots = properties; bar = mean)')
plt.tight_layout(); save('Figure_9_conformal.png')

# ---------------------------------------------------------------- Fig 10 interpretability
rows = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, 'exp', 'exp_interp', '*.json'))]
from exp_interp import BANDS, GROUP
fig, axs = plt.subplots(3, 1, figsize=(8.6, 6.6), sharex=True, gridspec_kw={'height_ratios': [0.8, 1, 1]})
spec = X.mean(0)
gap = np.where(np.diff(wnb) > 20)[0]
def gp(v):
    xs, ys = np.insert(wnb.astype(float), gap + 1, np.nan), np.insert(np.asarray(v, float), gap + 1, np.nan)
    return xs, ys
axs[0].plot(*gp(spec), color=INK2, lw=1)
axs[0].set_ylabel('Mean preprocessed\nspectrum (SD units)')
for ax, t, tg in [(axs[1], 'OrgC_pct', '(b)'), (axs[2], 'pH', '(c)')]:
    R = [r for r in rows if r['target'] == t]
    for k, c in [('attention', ORANGE), ('occlusion', BLUE), ('IG', AQUA), ('VIP', VIOLET)]:
        v = np.mean([np.array(r['profiles'][k]) for r in R], 0)
        ax.plot(*gp(v / v.max()), color=c, lw=1.4 if k == 'attention' else 1.0, label=k)
    for hi, lo in BANDS[GROUP[t]]:
        ax.axvspan(lo, hi, color='#eeeeea', zorder=0)
    ax.set_ylabel(f'{SHORT[t]}\nrelative importance'); tag(ax, tg)
for hi, lo in BANDS['organic'] + BANDS['carbonate']:
    pass
axs[0].axvspan(2250, 2450, color='#f6d8c4', zorder=0); axs[0].text(2350, axs[0].get_ylim()[1] * 0.85, 'CO$_2$\nmasked', ha='center', fontsize=6.5)
tag(axs[0], '(a)')
axs[1].legend(ncol=4, fontsize=7, loc='upper left')
axs[2].set_xlabel('Wavenumber (cm$^{-1}$)'); axs[2].set_xlim(4000, 600)
plt.tight_layout(); save('Figure_10_interpretability.png')
print('figures done')
