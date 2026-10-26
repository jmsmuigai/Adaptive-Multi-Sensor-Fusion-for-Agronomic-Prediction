"""Figure 1 -- study area, sampling structure and spectra (revised)."""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, OUT, AEZ_NAME, snv
import common

X, wnb, meta, Y = load()
Xr, wn = common._C['Xraw'], common._C['wn']
COL = {'Aw': '#2a78d6', 'Am': '#6250d6', 'BSh': '#eda100', 'BWh': '#e34948', 'C': '#1baf7a'}
MK = {'Kenya': 'o', 'Ethiopia': 's', 'Tanzania': '^', 'Uganda': 'D'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.6, 'legend.frameon': False})
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={'width_ratios': [1, 1.25, 1.2]})
ax = axs[0]
s = meta.groupby('SiteKey').agg(lat=('Latitude', 'mean'), lon=('Longitude', 'mean'), n=('SSN', 'size'),
                                c=('Country', 'first'), z=('AEZ', 'first'))
for k, r in s.iterrows():
    ax.scatter(r.lon, r.lat, s=18 + r.n * 0.6, color=COL[r.z], marker=MK[r.c], edgecolor='white', linewidth=0.7, zorder=3)
for c, m in MK.items():
    ax.scatter([], [], marker=m, color='#8a8984', label=c, s=28)
ax.axhline(0, color='#b9b8b2', lw=0.8, ls=':')
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.text(-0.14, 1.03, '(a)', transform=ax.transAxes, fontweight='bold', fontsize=10)
ax = axs[1]
s2 = s.sort_values(['z', 'n'], ascending=[True, False])
y = np.arange(len(s2))
ax.barh(y, s2.n, color=[COL[z] for z in s2.z], height=0.7)
ax.set_yticks(y); ax.set_yticklabels([f"{k.split('_')[1]} ({k.split('_')[0][:3]})" for k in s2.index], fontsize=6.8)
ax.invert_yaxis(); ax.set_xlabel('Samples per sentinel site'); ax.grid(axis='y', visible=False)
ax.text(-0.42, 1.03, '(b)', transform=ax.transAxes, fontweight='bold', fontsize=10)
ax = axs[2]
S = snv(Xr)
for z, c in COL.items():
    ax.plot(wn, S[meta.AEZ.values == z].mean(0), color=c, lw=1.2, label=z)
ax.axvspan(2250, 2450, color='#f6d8c4', zorder=0)
ax.text(2350, ax.get_ylim()[1] * 0.92, 'CO$_2$\nremoved', ha='center', fontsize=6.5)
ax.set_xlim(4000, 600); ax.set_xlabel('Wavenumber (cm$^{-1}$)'); ax.set_ylabel('Mean SNV absorbance')
ax.legend(fontsize=7, loc='lower left')
ax.text(-0.14, 1.03, '(c)', transform=ax.transAxes, fontweight='bold', fontsize=10)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
h = [Patch(color=c, label=AEZ_NAME[z]) for z, c in COL.items()] + \
    [Line2D([], [], marker=m, ls='', color='#8a8984', label=c) for c, m in MK.items()]
plt.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(handles=h, loc='lower center', ncol=5, fontsize=7.2, bbox_to_anchor=(0.5, 0.0))
plt.savefig(os.path.join(OUT, 'fig', 'Figure_1_study_area.png'), dpi=300, bbox_inches='tight', facecolor='white')
print('ok')
