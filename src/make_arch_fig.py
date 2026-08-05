"""make_arch_fig.py -- schematic of the proposed architecture (Figure 2)."""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures_ea')
os.makedirs(FIG, exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')

C_IN, C_CONV, C_ATT, C_RBF, C_OUT = '#dbe6f3', '#c8e0cd', '#f6dcc8', '#e8d3e8', '#f4cccc'

def box(x, y, w, h, txt, fc, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                                fc=fc, ec='#44546a', lw=1.1))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', linespacing=1.35)

def arrow(x1, y1, x2, y2, style='-|>'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=1.2, color='#44546a',
                                 shrinkA=1, shrinkB=1))

# --- spectral branch
box(1, 27, 13, 11, 'MIR spectrum\n1,749 bands\n601.7–4001.6 cm$^{-1}$', C_IN, bold=True)
box(16.5, 27, 12.5, 11, 'Preprocessing\nSG 1st derivative\nSNV · binning', C_IN)
box(31.5, 29.5, 12, 8, 'Conv1D + ReLU\nMaxPool', C_CONV, bold=True)
box(46, 29.5, 12, 8, 'Conv1D + ReLU\n$\\mathbf{H}\\in\\mathbb{R}^{p\\times C}$', C_CONV, bold=True)
box(60.5, 27, 15, 13,
    'DAFM\nattention over\nspectral positions\n'
    r'$\alpha_j=\mathrm{softmax}(e_j)$' + '\n' + r'$\mathbf{z}_s=\sum_j\alpha_j\mathbf{h}_j$',
    C_ATT, fs=8, bold=True)

# --- ancillary branch
box(1, 6, 13, 9, 'Ancillary inputs\ndepth · position', C_IN, bold=True)
box(31.5, 6.5, 12, 8, 'Dense encoder\n$\\mathbf{z}_c$', C_CONV)

# --- gate + head
box(60.5, 8, 15, 11, 'Modality gate\n'
    r'$\mathbf{g}=\mathrm{softmax}(\cdot)$' + '\nmeasured\ncontribution', C_ATT, fs=8, bold=True)
box(79, 18, 8.5, 12, 'Fusion\n$\\mathbf{z}$', C_ATT, bold=True)
box(89.5, 18, 9.5, 12,
    'RBF head\n$\\phi_i=e^{-\\gamma_i p_i^2}$\n$\\hat{y}$', C_RBF, fs=8.5, bold=True)

# --- arrows
arrow(14, 32.5, 16.5, 32.5); arrow(29, 32.5, 31.5, 33.5)
arrow(43.5, 33.5, 46, 33.5); arrow(58, 33.5, 60.5, 33.5)
arrow(14, 10.5, 31.5, 10.5); arrow(43.5, 10.5, 60.5, 13.5)
arrow(75.5, 33.5, 83.2, 30);  arrow(75.5, 13.5, 83.2, 18)
arrow(87.5, 24, 89.5, 24)

ax.text(50, 43.4, 'End-to-end differentiable: gradients flow from $\\hat{y}$ back to the '
        'convolutional kernels', ha='center', fontsize=9, style='italic', color='#44546a')
ax.add_patch(FancyArrowPatch((94, 31.5), (37.5, 39.6), connectionstyle='arc3,rad=0.22',
                             arrowstyle='-|>', mutation_scale=11, lw=1.1,
                             ls=(0, (4, 3)), color='#8a8a8a'))
ax.text(2, 40.5, 'Spectral branch', fontsize=9.5, fontweight='bold', color='#2f5597')
ax.text(2, 17.5, 'Ancillary branch', fontsize=9.5, fontweight='bold', color='#2f5597')
fig.savefig(os.path.join(FIG, 'Figure_2_architecture.png'), dpi=300, bbox_inches='tight')
print('wrote Figure_2_architecture.png')
