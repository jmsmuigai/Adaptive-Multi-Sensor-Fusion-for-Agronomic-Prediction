"""
common.py -- shared data, preprocessing, fold designs and metrics for the
revised (round-2) analysis requested by the JAGST reviewer.

Key changes relative to the round-1 pipeline
  * every fold of every design is a test fold; hyper-parameters are selected
    INSIDE each training fold (nested, group-aware)
  * early stopping uses a GROUP-AWARE inner validation split (never the test fold)
  * all models receive the same inputs (pre-processed spectra + depth flag);
    coordinates are only used in an explicit ablation
  * CO2 region (2450-2250 cm-1) masked; preprocessing alternatives tested
  * soil-spectroscopy metrics: RPD, RPIQ, Lin's CCC, RMSE/SD_train, MAE skill
"""
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

SEED = 20260804
TARGETS = ['pH', 'OrgC_pct', 'TotN_pct', 'P_Olsen', 'Ca_exch', 'Mg_exch',
           'K_exch', 'eCEC', 'S_total', 'Fe_total', 'Mn_total', 'Zn_total',
           'Cu_total']
LABEL = {'pH': 'pH', 'OrgC_pct': 'Organic C', 'TotN_pct': 'Total N',
         'P_Olsen': 'Olsen P', 'Ca_exch': 'Exch. Ca', 'Mg_exch': 'Exch. Mg',
         'K_exch': 'Exch. K', 'eCEC': 'eCEC', 'S_total': 'Total S',
         'Fe_total': 'Total Fe', 'Mn_total': 'Total Mn', 'Zn_total': 'Total Zn',
         'Cu_total': 'Total Cu'}

# Koppen-Geiger zone of each sentinel site (majority class of its samples,
# kgcpy lookup); used for the leave-AEZ-out design.
SITE_AEZ = {
    'Ethiopia_Dambidolo': 'Aw', 'Tanzania_Bukwaya': 'Aw', 'Tanzania_Kiberashi': 'Aw',
    'Tanzania_Kidatu': 'Aw', 'Uganda_Hoima': 'Aw',
    'Uganda_Namasuba': 'Am',
    'Kenya_Marafa': 'BSh', 'Tanzania_Chinyanghuku': 'BSh', 'Tanzania_Itende': 'BSh',
    'Tanzania_Pandambili': 'BSh',
    'Kenya_Mbalambala': 'BWh',
    'Ethiopia_Kutaber': 'C', 'Ethiopia_Mega': 'C', 'Ethiopia_Merar': 'C',
    'Kenya_Morijo': 'C', 'Tanzania_Mbinga': 'C',
}
AEZ_NAME = {'Aw': 'Tropical savanna (Aw)', 'Am': 'Tropical monsoon (Am)',
            'BSh': 'Hot semi-arid (BSh)', 'BWh': 'Hot arid (BWh)',
            'C': 'Temperate highland (Cfb/Cwb/Csb/Cwa)'}

CO2_MASK = (2250.0, 2450.0)

_C = {}


def load():
    if 'X' not in _C:
        z = np.load(os.path.join(DATA, 'ea_spectra.npz'))
        _C['Xraw'] = z['X'].astype(float)
        _C['wn'] = z['wavenumbers'].astype(float)
        m = pd.read_csv(os.path.join(DATA, 'ea_meta.csv'))
        m['AEZ'] = m.SiteKey.map(SITE_AEZ)
        m['PlotKey'] = m.ClusterKey + '_' + m.Plot.astype(str)
        _C['meta'] = m
        _C['Y'] = pd.read_csv(os.path.join(DATA, 'ea_targets.csv'))
        Xp, wnb = static_prep(_C['Xraw'], _C['wn'], **PREP_DEFAULT)
        _C['X'] = Xp
        _C['wnb'] = wnb
    return _C['X'], _C['wnb'], _C['meta'], _C['Y']


# ----------------------------------------------------------- preprocessing
def savgol(X, window=11, poly=2, deriv=0):
    half = window // 2
    idx = np.arange(-half, half + 1)
    A = np.vander(idx, poly + 1, increasing=True)
    coef = np.linalg.pinv(A)[deriv] * (math.factorial(deriv) if deriv else 1.0)
    Xp = np.pad(X, ((0, 0), (half, half)), mode='edge')
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = Xp[:, j:j + window] @ coef
    return out


def snv(X):
    m = X.mean(1, keepdims=True)
    s = X.std(1, keepdims=True)
    s[s < 1e-12] = 1.0
    return (X - m) / s


def static_prep(X, wn, deriv=1, window=11, binf=4, order='sg_snv', mask_co2=True):
    """Sample-wise (row-wise) operations only: no statistic is shared between
    samples, so applying them to the full matrix cannot leak information."""
    if order == 'sg_snv':
        Z = snv(savgol(X, window, 2, deriv)) if deriv is not None else snv(X)
    elif order == 'snv_sg':
        Z = savgol(snv(X), window, 2, deriv)
    elif order == 'snv':
        Z = snv(X)
    else:
        raise ValueError(order)
    keep = np.ones(len(wn), bool)
    if mask_co2:
        keep &= ~((wn >= CO2_MASK[0]) & (wn <= CO2_MASK[1]))
    Z, w = Z[:, keep], wn[keep]
    n, L = Z.shape
    L2 = (L // binf) * binf
    Zb = Z[:, :L2].reshape(n, L2 // binf, binf).mean(2)
    wb = w[:L2].reshape(L2 // binf, binf).mean(1)
    return Zb, wb


PREP_DEFAULT = dict(deriv=1, window=11, binf=4, order='sg_snv', mask_co2=True)


class XScaler:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class YPrep:
    """log1p decision and standardisation fitted on training rows only."""

    def fit(self, y):
        self.log = bool(pd.Series(y).skew() > 2.0)
        t = np.log1p(np.maximum(y, 0)) if self.log else y
        self.mu, self.sd = float(np.mean(t)), float(np.std(t)) or 1.0
        return self

    def fwd(self, y):
        t = np.log1p(np.maximum(y, 0)) if self.log else y
        return (t - self.mu) / self.sd

    def inv(self, z):
        t = np.asarray(z) * self.sd + self.mu
        return np.expm1(t) if self.log else t

    def to_log(self, z):
        """standardised -> fitting scale (log1p or raw)"""
        return np.asarray(z) * self.sd + self.mu


def ancillary(meta, idx, coords=False):
    m = meta.iloc[idx]
    cols = [m.is_topsoil.values.astype(float)]
    if coords:
        cols += [m.Latitude.values, m.Longitude.values]
    return np.column_stack(cols).astype(float)


# ------------------------------------------------------------- fold designs
NFOLDS = {'random': 5, 'cluster': 5, 'site': 8, 'country': 4, 'aez': 5}


def group_key(design, meta):
    return {'random': None,
            'cluster': meta.ClusterKey.values,
            'site': meta.SiteKey.values,
            'country': meta.Country.values,
            'aez': meta.AEZ.values}[design]


def _greedy(key, k):
    uniq, counts = np.unique(key, return_counts=True)
    k = min(k, len(uniq))
    order = np.argsort(-counts)
    load_ = np.zeros(k, dtype=int)
    assign = {}
    for gi in order:
        f = int(np.argmin(load_))
        assign[uniq[gi]] = f
        load_[f] += counts[gi]
    fo = np.array([assign[g] for g in key])
    return [np.where(fo == f)[0] for f in range(k)]


def outer_folds(design, meta):
    if design == 'random':
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(meta))
        return [np.sort(f) for f in np.array_split(idx, 5)]
    return _greedy(group_key(design, meta), NFOLDS[design])


def inner_groups(design, meta, tr):
    """Grouping used for the inner (tuning / early-stopping) split. It is the
    same spatial unit as the outer design where possible; for country / AEZ
    designs, sites (the next unit down) are used because only 3-4 outer
    groups remain in the training data."""
    m = meta.iloc[tr]
    if design == 'random':
        return None
    if design == 'cluster':
        return m.ClusterKey.values
    return m.SiteKey.values


def inner_split(design, meta, tr, frac=0.2, seed=SEED):
    """One group-disjoint validation split inside the training fold, used for
    early stopping and for choosing among neural configurations."""
    rng = np.random.default_rng(seed)
    g = inner_groups(design, meta, tr)
    if g is None:
        p = rng.permutation(len(tr))
        nv = int(round(frac * len(tr)))
        return tr[p[nv:]], tr[p[:nv]]
    uniq, cnt = np.unique(g, return_counts=True)
    order = rng.permutation(len(uniq))
    target = frac * len(tr)
    val_groups, s = [], 0
    for i in order:
        if s >= target:
            break
        if s + cnt[i] > 1.8 * target and s > 0:
            continue
        val_groups.append(uniq[i]); s += cnt[i]
    vmask = np.isin(g, val_groups)
    return tr[~vmask], tr[vmask]


def inner_kfold(design, meta, tr, k=3):
    """k group-disjoint inner folds (for the cheap models' grid search)."""
    g = inner_groups(design, meta, tr)
    if g is None:
        rng = np.random.default_rng(SEED)
        p = rng.permutation(len(tr))
        parts = np.array_split(p, k)
        return [(np.setdiff1d(np.arange(len(tr)), q), q) for q in parts]
    fo = _greedy(g, k)
    return [(np.setdiff1d(np.arange(len(tr)), f), f) for f in fo]


# ----------------------------------------------------------------- metrics
def lin_ccc(y, p):
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = np.mean((y - my) * (p - mp))
    return float(2 * cov / (vy + vp + (my - mp) ** 2)) if (vy + vp) > 0 else np.nan


def metrics(y, p, y_train=None):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    out = {'R2': 1 - float(np.sum(e ** 2)) / sst if sst > 0 else np.nan,
           'RMSE': rmse, 'MAE': float(np.mean(np.abs(e))), 'bias': float(e.mean()),
           'RPD': float(y.std(ddof=1) / rmse) if rmse > 0 else np.nan,
           'RPIQ': float((np.percentile(y, 75) - np.percentile(y, 25)) / rmse) if rmse > 0 else np.nan,
           'CCC': lin_ccc(y, p), 'n': int(len(y))}
    if y_train is not None:
        sdt = float(np.std(y_train, ddof=1))
        out['nRMSE_sdtrain'] = rmse / sdt if sdt > 0 else np.nan
        mae0 = float(np.mean(np.abs(y - np.mean(y_train))))
        out['MAE_skill'] = 1 - out['MAE'] / mae0 if mae0 > 0 else np.nan
    return out
