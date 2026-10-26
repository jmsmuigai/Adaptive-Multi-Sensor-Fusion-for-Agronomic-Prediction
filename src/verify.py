"""verify.py -- independent re-computation of headline numbers directly from the
per-fit records (not from results.py), compared with numbers.json."""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, TARGETS, OUT, outer_folds

X, wnb, meta, Y = load()
N = json.load(open(os.path.join(OUT, 'numbers.json')))
fails = 0


def check(name, got, want, tol=0.002):
    global fails
    ok = abs(got - want) <= tol
    fails += (not ok)
    print(f"{'OK ' if ok else 'BAD'} {name}: recomputed {got:.4f} vs reported {want:.4f}")


def pooled_r2(design, model):
    vals = []
    folds = outer_folds(design, meta)
    for t in TARGETS:
        y = Y[t].values
        idx, pr = [], []
        for f in range(len(folds)):
            r = json.load(open(os.path.join(OUT, 'parts', f'{design}__{t}__{model}__f{f}.json')))
            P = np.atleast_2d(np.array(r['pred']))
            tr = np.setdiff1d(np.where(np.isfinite(y))[0], folds[f])
            P = np.clip(P, y[tr].min(), y[tr].max())
            idx.append(np.array(r['test_idx'])); pr.append(P)
        idx = np.concatenate(idx); P = np.concatenate(pr, 1); yy = y[idx]
        vals.append(np.mean([1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum() for p in P]))
    return float(np.mean(vals))


check('GPR leave-site-out mean R2', pooled_r2('site', 'GPR'), N['best_site_r2'])
check('Hybrid leave-site-out mean R2', pooled_r2('site', 'Hybrid'), N['hyb_site_r2'])
check('Hybrid leave-country-out mean R2', pooled_r2('country', 'Hybrid'), N['hyb_country_r2'])
check('GPR leave-country-out mean R2', pooled_r2('country', 'GPR'), N['best_country_r2'])
check('PLSR leave-site-out mean R2', pooled_r2('site', 'PLSR'), N['plsr_site_r2'])
# conformal coverage, site, group calibration, averaged over models
rows = []
for base in [os.path.join(OUT, 'vm', 'exp'), os.path.join(OUT, 'exp')]:
    for f in glob.glob(os.path.join(base, 'exp_conformal', 'site__*.json')):
        rows += json.load(open(f))
C = pd.DataFrame(rows)
cov = C[C.cal == 'group'].groupby('model').apply(lambda g: (g.coverage * g.n_test).sum() / g.n_test.sum()).mean()
check('conformal coverage site/group (mean of models)', cov, N['cf_site_group_mean'])
# data facts
check('Olsen P % below 2 mg/kg', float((Y.P_Olsen.dropna() < 2).mean() * 100), N['p_below2'], 0.01)
check('n samples', len(meta), 1310, 0)
print('FAILURES:', fails)
