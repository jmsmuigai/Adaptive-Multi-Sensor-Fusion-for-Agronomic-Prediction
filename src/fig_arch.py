"""Figure 2 -- revised architecture diagram."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'fig')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, ax = plt.subplots(figsize=(10, 4.3), dpi=300)
ax.set_xlim(0, 101); ax.set_ylim(0, 44); ax.axis('off')

C = {'in': '#E8EEF4', 'conv': '#CFE0EE', 'att': '#F6D8C4', 'anc': '#DDEBD5', 'gate': '#EDE1F2',
     'head': '#F9E7B5', 'out': '#E8EEF4'}


def box(x, y, w, h, title, sub, col, ls='-'):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                       fc=col, ec='#333333', lw=0.9, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 2.2, title, ha='center', va='top', fontsize=9.5, fontweight='bold')
    ax.text(x + w / 2, y + h - 6.0, sub, ha='center', va='top', fontsize=7.6, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color='#333333', mutation_scale=10))


box(1, 26, 13, 15, 'MIR spectrum', '601-4002 cm$^{-1}$\nSG 1st deriv. + SNV\nCO$_2$ band removed\n415 bins', C['in'])
box(17.5, 26, 17, 15, '1D-CNN', 'Conv(k$_1$, s$_1$) + ReLU\nMax-pool (2)\nConv(k$_2$) + ReLU\n+ positional embedding', C['conv'])
box(38, 26, 16, 15, 'DAFM attention', 'e$_j$ = v$^T$tanh(W h$_j$+b)\n$\\alpha$ = softmax(e)\nz$_s$ = $\\Sigma_j\\,\\alpha_j$ h$_j$\n(broad windows: Sec. 2.4.1)', C['att'])
box(1, 4, 13, 14, 'Ancillary', 'sampling depth\n(top / subsoil)\n[coordinates only\nin ablation]', C['anc'], ls='--')
box(17.5, 4, 17, 14, 'Dense encoder', 'z$_c$ = ReLU(W$_c$a + b$_c$)', C['anc'])
box(57.5, 14, 17, 16, 'Modality gate', 'g = softmax(W$_g$[z$_s$;z$_c$])\nz = [g$_1$z$_s$; g$_2$z$_c$]\n(descriptive only)', C['gate'])
box(77.5, 14, 15, 16, 'RBF head', 'u = W$_r$z + b$_r$\n$\\phi_i$ = exp(-$\\gamma_i$||u - c$_i$||$^2$)\nlearned centres c$_i$\n$\\hat{y}$ = w$_o^T\\phi$ + b$_o$', C['head'])
box(94.5, 17, 6, 10, '$\\hat{y}$', 'back-\ntrans-\nformed', C['out'])

arrow(14.6, 33.5, 17.0, 33.5); arrow(35.1, 33.5, 37.5, 33.5)
arrow(14.6, 11, 17.0, 11)
arrow(54.6, 33.5, 60, 30.5); arrow(35.1, 11, 60, 14.0)
arrow(75.1, 22, 77.0, 22); arrow(93.1, 22, 94.0, 22)
ax.text(64, 40.5, 'Ablations replace one component at a time:', fontsize=7.8, style='italic')
ax.text(64, 37.2, 'head: linear | dense | fixed-centre Gaussian;  pooling: mean | re-weight', fontsize=7.4)
ax.text(64, 34.4, 'no positional embedding;  gate -> concat;  spectra only;  + coordinates', fontsize=7.4)
plt.savefig(os.path.join(OUT, 'Figure_2_architecture.png'), bbox_inches='tight', facecolor='white')
print('ok')
