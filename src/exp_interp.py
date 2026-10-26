"""
exp_interp.py -- validation of the attention-based interpretation
(reviewer section 2.4).

For the leave-site-out hybrid models saved in the main run (first seed):
  attention  : DAFM weights per conv position spread onto the input bins each
               position covers (its receptive field)
  occlusion  : increase in test MSE when a 8-bin (~62 cm-1) window is set to
               the training mean, window stride 4 bins
  IG         : mean |integrated gradients| (zero baseline = training mean,
               32 steps)
  VIP        : PLSR variable importance in projection, PLSR fitted on the same
               training fold with the component number chosen in the main run
Agreement: Spearman correlation between the four importance profiles.
Overlap test: share of importance mass inside literature band windows,
against a null of 2000 random circular shifts of the profile.
Contribution of the ancillary branch: permutation importance (depth
permuted in the test fold) and the spectra-only ablation, instead of the gate.
"""
import json
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, ancillary, outer_folds, inner_split, XScaler, YPrep, OUT
from nets import HybridNet
from sklearn.cross_decomposition import PLSRegression

PARTS = os.path.join(OUT, 'parts')
# literature band windows (cm-1); Nguyen et al. 1991; Janik et al. 1998;
# Parikh et al. 2014; Soriano-Disla et al. 2014
BANDS = {
    'organic': [(2950, 2840), (1750, 1700), (1660, 1580), (1560, 1520), (1470, 1370)],
    'carbonate': [(2620, 2480), (1820, 1780), (1500, 1400), (890, 860)],
    'mineral': [(3710, 3600), (930, 900), (1100, 980), (800, 770)],
}
GROUP = {'OrgC_pct': 'organic', 'TotN_pct': 'organic', 'Ca_exch': 'carbonate',
         'pH': 'carbonate', 'eCEC': 'mineral', 'Fe_total': 'mineral'}


def vip(pls):
    t = pls.x_scores_; w = pls.x_weights_; q = pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    tot = s.sum()
    wn = (w / np.linalg.norm(w, axis=0)) ** 2
    return np.sqrt(p * (wn @ s).ravel() / tot)


def in_bands(wnb, bands):
    m = np.zeros(len(wnb), bool)
    for hi, lo in bands:
        m |= (wnb <= hi) & (wnb >= lo)
    return m


def shift_test(imp, mask, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    imp = imp / imp.sum()
    obs = imp[mask].sum()
    null = np.array([np.roll(imp, rng.integers(1, len(imp)))[mask].sum() for _ in range(B)])
    return float(obs), float(null.mean()), float((1 + (null >= obs).sum()) / (B + 1))


def job(args):
    t, f = args
    X, wnb, meta, Y = load()
    A = ancillary(meta, np.arange(len(meta)))
    y = Y[t].values.astype(float)
    ok = np.isfinite(y)
    folds = outer_folds('site', meta)
    te = np.intersect1d(folds[f], np.where(ok)[0])
    tr = np.setdiff1d(np.where(ok)[0], folds[f])
    trn, va = inner_split('site', meta, tr)
    sc = XScaler().fit(X[trn]); asc = XScaler().fit(A[trn]); yp = YPrep().fit(y[trn])
    part = json.load(open(os.path.join(PARTS, f'site__{t}__Hybrid__f{f}.json')))
    cfg = part['params']
    net = HybridNet(X.shape[1], A.shape[1], **cfg)
    net.load_state_dict(torch.load(os.path.join(PARTS, f'site__{t}__Hybrid__f{f}.pt')))
    net.eval()
    Xt = torch.as_tensor(sc(X[te]), dtype=torch.float32)
    At = torch.as_tensor(asc(A[te]), dtype=torch.float32)
    yt = yp.fwd(y[te])
    L = X.shape[1]
    net._store = True
    with torch.no_grad():
        base = net(Xt, At).numpy()
    alpha = net.cache['alpha'].mean(0).numpy()
    net._store = False
    mse0 = np.mean((base - yt) ** 2)
    # attention spread over receptive fields
    att = np.zeros(L); cnt = np.zeros(L)
    for j, a in enumerate(alpha):
        s0 = j * net.jump; s1 = min(L, s0 + net.rf)
        att[s0:s1] += a; cnt[s0:s1] += 1
    att = att / np.maximum(cnt, 1)
    # occlusion
    occ = np.zeros(L); oc = np.zeros(L)
    for s0 in range(0, L - 8 + 1, 4):
        Xo = Xt.clone(); Xo[:, s0:s0 + 8] = 0.0
        with torch.no_grad():
            d = np.mean((net(Xo, At).numpy() - yt) ** 2) - mse0
        occ[s0:s0 + 8] += max(d, 0); oc[s0:s0 + 8] += 1
    occ = occ / np.maximum(oc, 1)
    # integrated gradients
    steps = 32
    ig = torch.zeros_like(Xt)
    for k in range(1, steps + 1):
        xk = (Xt * k / steps).requires_grad_(True)
        out = net(xk, At).sum()
        g, = torch.autograd.grad(out, xk)
        ig += g
    ig = (Xt * ig / steps).abs().mean(0).numpy()
    # PLSR VIP
    pp = json.load(open(os.path.join(PARTS, f'site__{t}__PLSR__f{f}.json')))['params']
    sx = XScaler().fit(X[tr])
    pls = PLSRegression(n_components=int(pp['n_components']), scale=False).fit(sx(X[tr]), YPrep().fit(y[tr]).fwd(y[tr]))
    v = vip(pls)
    # permutation importance of the ancillary (depth) input
    rng = np.random.default_rng(0)
    dm = []
    for _ in range(20):
        Ap = At[torch.as_tensor(rng.permutation(len(te)))]
        with torch.no_grad():
            dm.append(np.mean((net(Xt, Ap).numpy() - yt) ** 2) - mse0)
    Xp = Xt[torch.as_tensor(rng.permutation(len(te)))]
    with torch.no_grad():
        dspec = np.mean((net(Xp, At).numpy() - yt) ** 2) - mse0
    prof = {'attention': att, 'occlusion': occ, 'IG': ig, 'VIP': v}
    res = {'target': t, 'fold': f, 'rf_bins': int(net.rf), 'jump_bins': int(net.jump),
           'perm_depth_dMSE': float(np.mean(dm)), 'perm_spec_dMSE': float(dspec),
           'mse0': float(mse0), 'profiles': {k: (pv / pv.sum()).round(6).tolist() for k, pv in prof.items()},
           'gate_spec': float(np.mean([s.get('gate_spec', np.nan) for s in part['seeds']]))}
    ks = list(prof)
    res['spearman'] = {f'{a}~{b}': float(spearmanr(prof[a], prof[b]).correlation)
                       for i, a in enumerate(ks) for b in ks[i + 1:]}
    if t in GROUP:
        mask = in_bands(wnb, BANDS[GROUP[t]])
        res['overlap'] = {k: shift_test(pv, mask, seed=f) for k, pv in prof.items()}
    return res


def jobs():
    return [(t, f) for t in TARGETS for f in range(8)]


if __name__ == '__main__':
    jobs = [(t, f) for t in TARGETS for f in range(8)]
    with Pool(2) as p:
        rows = p.map(job, jobs, chunksize=1)
    json.dump(rows, open(os.path.join(OUT, 'exp_interp.json'), 'w'))
    sp = pd.DataFrame([dict(target=r['target'], **r['spearman']) for r in rows])
    print(sp.groupby('target').mean().round(2))
    ov = [dict(target=r['target'], method=k, obs=v[0], null=v[1], p=v[2])
          for r in rows if 'overlap' in r for k, v in r['overlap'].items()]
    print(pd.DataFrame(ov).groupby(['target', 'method']).agg(['mean', 'median']).round(3))
