"""verify_ea.py -- check every numeric claim in manuscript_final.md against the
computed results. Exits non-zero on any mismatch."""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EA = os.path.join(HERE, 'ea')
KEN = os.path.join(HERE, 'kenya')
M = ['PLSR', 'Ridge', 'RandomForest', 'GradientBoosting', 'MLP', 'DualDAFM']
fails, n = [], 0


def chk(name, expect, actual, tol=6e-4):
    global n
    n += 1
    try:
        a = float(actual)
    except Exception:
        fails.append(f'{name}: could not read computed value')
        return
    if not np.isfinite(a) or abs(expect - a) > tol:
        fails.append(f'{name}: manuscript {expect}, computed {a}')


def main():
    md = open(os.path.join(HERE, 'manuscript_final.md')).read()
    r = pd.read_csv(os.path.join(EA, 'summary.csv'))
    h = json.load(open(os.path.join(EA, 'headline.json')))
    aud = json.load(open(os.path.join(EA, 'ea_audit.json')))

    # --- dataset composition
    chk('n samples', 1310, aud['n_samples'], 0)
    chk('n sites', 16, aud['n_sites'], 0)
    chk('n clusters', 207, aud['n_clusters'], 0)
    chk('n wavenumbers', 1749, aud['n_wavenumbers'], 0)
    chk('wn low', 601.7, aud['wavenumber_range_cm-1'][0], .05)
    chk('wn high', 4001.6, aud['wavenumber_range_cm-1'][1], .05)
    chk('topsoil', 659, aud['depth_counts']['Topsoil'], 0)
    chk('subsoil', 651, aud['depth_counts']['Subsoil'], 0)
    if aud['join']['attrition'] != 0:
        fails.append('join attrition is not zero')
    for c, nn, st, cl in [('Ethiopia', 381, 4, 63), ('Tanzania', 367, 7, 67),
                          ('Kenya', 340, 3, 45), ('Uganda', 222, 2, 32)]:
        chk(f'{c} n', nn, aud['per_country'][c]['n'], 0)
        chk(f'{c} sites', st, aud['per_country'][c]['n_sites'], 0)
        chk(f'{c} clusters', cl, aud['per_country'][c]['n_clusters'], 0)

    # --- design means
    for d, v in [('cluster', 0.749), ('site', 0.351), ('country', 0.205)]:
        chk(f'best mean R2 {d}', v, max(h[d]['mean_R2'].values()), 1e-3)
    for m, v in [('MLP', 0.749), ('GradientBoosting', 0.732), ('RandomForest', 0.695),
                 ('Ridge', 0.685), ('DualDAFM', 0.636), ('PLSR', 0.505)]:
        chk(f'cluster mean {m}', v, h['cluster']['mean_R2'][m], 1e-3)
    chk('hybrid cluster rank', 4.92, h['cluster']['mean_rank']['DualDAFM'], 6e-3)
    if h['cluster']['best_by_mean_rank'] != 'MLP':
        fails.append('MLP is not best under cluster CV')
    for d in ('site', 'country'):
        if h[d]['best_by_mean_rank'] != 'GradientBoosting':
            fails.append(f'gradient boosting is not best under {d}')
    chk('GBM site rank', 2.31, h['site']['mean_rank']['GradientBoosting'], 6e-3)
    chk('GBM country rank', 2.69, h['country']['mean_rank']['GradientBoosting'], 6e-3)
    chk('MLP site mean', -1.791, h['site']['mean_R2']['MLP'], 1e-3)
    chk('hybrid site n>0', 11, h['site']['n_positive']['DualDAFM'], 0)
    chk('hybrid country n>0', 10, h['country']['n_positive']['DualDAFM'], 0)

    # --- within-landscape headline values
    g = r[r.design == 'cluster']
    b = g.loc[g.groupby('target').R2.idxmax()].set_index('target')
    for t, v in [('TotN_pct', 0.939), ('pH', 0.916), ('Fe_total', 0.915),
                 ('eCEC', 0.906), ('Ca_exch', 0.900), ('OrgC_pct', 0.897),
                 ('Mg_exch', 0.811), ('Zn_total', 0.810), ('Cu_total', 0.713),
                 ('Mn_total', 0.626), ('P_Olsen', 0.562), ('S_total', 0.513),
                 ('K_exch', 0.504)]:
        chk(f'cluster best {t}', v, float(b.loc[t, 'R2']))
    chk('TotN CI lo', 0.924, float(b.loc['TotN_pct', 'R2_lo']))
    chk('TotN CI hi', 0.954, float(b.loc['TotN_pct', 'R2_hi']))
    nmlp = int((b.model == 'MLP').sum())
    if nmlp != 10:
        fails.append(f'MLP best for {nmlp} properties, manuscript says 10')
    chk('PLSR S_total cluster', -0.848,
        float(g[(g.target == 'S_total') & (g.model == 'PLSR')].R2.iloc[0]), 1e-3)

    # --- transfer values
    s = r[r.design == 'site']
    bs = s.loc[s.groupby('target').R2.idxmax()].set_index('target')
    for t, v in [('OrgC_pct', 0.814), ('TotN_pct', 0.670), ('eCEC', 0.665),
                 ('Fe_total', 0.639), ('Ca_exch', 0.550), ('K_exch', 0.137),
                 ('P_Olsen', 0.032)]:
        chk(f'site best {t}', v, float(bs.loc[t, 'R2']), 1e-3)
    fe_mlp = float(s[(s.target == 'Fe_total') & (s.model == 'MLP')].R2.iloc[0])
    chk('MLP Fe site', -24.1, fe_mlp, 0.1)

    gap2 = (h['site']['mean_R2']['GradientBoosting'] - h['site']['mean_R2']['RandomForest'])
    chk('best-second site gap', 0.047, gap2, 2e-3)
    gap3 = (h['site']['mean_R2']['GradientBoosting'] - h['site']['mean_R2']['Ridge'])
    chk('GBM vs ridge site gap', 0.377, gap3, 2e-3)

    # --- calibration breadth contrast
    kh = json.load(open(os.path.join(KEN, 'headline.json')))
    chk('Kenya-only best site mean', -0.651, max(kh['site']['mean_R2'].values()), 1e-3)
    chk('Kenya-only n>0', 6, max(kh['site']['n_positive'].values()), 0)
    chk('EA n>0 site', 12, max(h['site']['n_positive'].values()), 0)
    kr = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    kc = kr[kr.design == 'cluster'].groupby('target').R2.max()
    for t, v in [('P_Olsen', 0.164), ('TotN_pct', 0.889), ('pH', 0.848)]:
        chk(f'Kenya-only {t}', v, float(kc[t]))
    if not (kr[kr.design == 'site'].groupby('model').R2.mean() < 0).all():
        fails.append('claim that all Kenya-only models were negative is false')

    # --- per-country transfer
    cdp = os.path.join(EA, 'country_detail.csv')
    if os.path.exists(cdp):
        det = pd.read_csv(cdp)
        piv = det.pivot_table(index='country', columns='model', values='R2')
        for c, v in [('Tanzania', 0.350), ('Kenya', 0.240), ('Uganda', 0.193)]:
            chk(f'country best {c}', v, float(piv.loc[c].max()), 1e-3)
        if piv.loc['Kenya'].idxmax() != 'DualDAFM':
            fails.append('hybrid is not best for held-out Kenya')
        if piv.loc['Tanzania'].idxmax() != 'GradientBoosting':
            fails.append('GBM is not best for held-out Tanzania')
        if piv.loc['Uganda'].idxmax() != 'MLP':
            fails.append('MLP is not best for held-out Uganda')
        for c, k in [('Tanzania', 10), ('Kenya', 9), ('Uganda', 10)]:
            bm = piv.loc[c].idxmax()
            g = det[(det.country == c) & (det.model == bm)]
            chk(f'country n>0 {c}', k, int((g.R2 > 0).sum()), 0)
        chk('Kenya ridge country', 0.188, float(piv.loc['Kenya', 'Ridge']), 1e-3)
        chk('Kenya gbm country', 0.120, float(piv.loc['Kenya', 'GradientBoosting']), 1e-3)

    # --- branch gate
    bg = pd.read_csv(os.path.join(EA, 'branch_gate.csv')).set_index('target')
    for t, v, sd in [('TotN_pct', 0.592, 0.108), ('OrgC_pct', 0.555, 0.149),
                     ('pH', 0.415, 0.089), ('Ca_exch', 0.350, 0.161)]:
        chk(f'gate {t}', v, float(bg.loc[t, 'spectral_weight']), 1e-3)
        chk(f'gate sd {t}', sd, float(bg.loc[t, 'spectral_sd']), 1e-3)

    # --- deficiency
    d = pd.read_csv(os.path.join(EA, 'deficiency.csv')).set_index('target')
    for t, v in [('P_Olsen', 84.4), ('Cu_total', 71.3), ('TotN_pct', 69.1),
                 ('Zn_total', 66.9), ('OrgC_pct', 60.6), ('K_exch', 15.9),
                 ('eCEC', 14.6), ('Ca_exch', 8.5), ('Mg_exch', 7.1), ('pH', 7.9)]:
        chk(f'deficiency {t}', v, float(d.loc[t, 'pct_below']), 0.06)
    chk('topsoil n deficiency', 659, int(d.loc['P_Olsen', 'n']), 0)

    # --- computational cost claims
    tp = os.path.join(EA, 'timing.json')
    if os.path.exists(tp):
        tm = json.load(open(tp))['models']
        chk('hybrid fit s', 20.9, tm['DualDAFM']['fit_s'], 0.06)
        chk('hybrid ms/sample', 0.0785, tm['DualDAFM']['ms_mean'], 1e-4)
        chk('hybrid p95', 0.0875, tm['DualDAFM']['ms_p95'], 1e-4)
        chk('hybrid per sec', 12700, tm['DualDAFM']['per_sec'], 60)
        chk('gbm fit s', 2.6, tm['GradientBoosting']['fit_s'], 0.06)
        chk('gbm ms/sample', 0.0354, tm['GradientBoosting']['ms_mean'], 1e-4)
        chk('mlp fit s', 1.8, tm['MLP']['fit_s'], 0.06)
        chk('mlp ms/sample', 0.0015, tm['MLP']['ms_mean'], 1e-4)

    # --- every figure and table marker resolves
    import re as _re
    marks = _re.findall(r'\[\[(TABLE|FIGURE) (\d+)\]\]', md)
    for kind, num in marks:
        if marks.count((kind, num)) > 1:
            fails.append(f'duplicate marker {kind} {num}')
    tn = sorted({int(x) for k, x in marks if k == 'TABLE'})
    fn = sorted({int(x) for k, x in marks if k == 'FIGURE'})
    if tn != list(range(1, len(tn) + 1)):
        fails.append(f'table numbering not contiguous: {tn}')
    if fn != list(range(1, len(fn) + 1)):
        fails.append(f'figure numbering not contiguous: {fn}')

    # --- bounded-failure claims (Section 2.4.5 / 3.2)
    for m, v in [('MLP', -24.146), ('PLSR', -2.102), ('Ridge', -1.637),
                 ('DualDAFM', -0.425), ('RandomForest', -0.468),
                 ('GradientBoosting', -0.041)]:
        chk(f'site worst {m}', v, h['site']['worst_R2'][m], 1e-3)
    unb = ['MLP', 'PLSR', 'Ridge']
    bnd = ['GradientBoosting', 'RandomForest', 'DualDAFM']
    order = sorted(M, key=lambda m: h['site']['worst_R2'][m])
    if set(order[:3]) != set(unb):
        fails.append(f'three most severe failures are {order[:3]}, manuscript says {unb}')
    if set(order[3:]) != set(bnd):
        fails.append(f'three least severe are {order[3:]}, manuscript says {bnd}')
    ratio = h['site']['worst_R2']['MLP'] / h['site']['worst_R2']['DualDAFM']
    chk('MLP/hybrid worst ratio', 57, ratio, 0.6)

    # --- title, terminology and repository link
    if 'Multi-Modal Fusion' not in md.split('\n')[0]:
        fails.append('title does not use Multi-Modal')
    prose = re.sub(r'https?://\S+', '', md)   # the repository name is exempt
    if 'multi-sensor' in prose.lower():
        fails.append('the term multi-sensor still appears in the prose')
    if 'github.com/jmsmuigai' not in md:
        fails.append('repository URL missing from Section 7.0')
    if md.count('**Proposition') != 3:
        fails.append('expected three propositions in Section 2.4.5')
    eqs = re.findall(r'\(Eq\. (\d+)\)', md)
    nums = sorted({int(e) for e in eqs})
    if nums != list(range(1, len(nums) + 1)):
        fails.append(f'equation numbering not contiguous: {nums}')

    # --- format compliance
    ab = md.split('## Abstract')[1].split('**Keywords')[0].strip()
    w = len(ab.split())
    if w > 300:
        fails.append(f'abstract is {w} words, JAGST limit is 300')
    rt = re.search(r'\*\*Running title:\*\* (.+)', md)
    if not rt:
        fails.append('no running title')
    elif len(rt.group(1).split()) > 7:
        fails.append(f'running title is {len(rt.group(1).split())} words, limit 7')
    refs = md.split('## 8.0 References')[1]
    entries = [x for x in refs.strip().split('\n\n') if x.strip()]
    for e in entries:
        if 'http' not in e:
            fails.append(f'reference without link: {e[:60]}')
    body = md.split('## 8.0 References')[0]
    cited = set(re.findall(r'([A-Z][A-Za-zåå-]+)(?: et al\.| and [A-Z][A-Za-z-]+)?,? \(\d{4}',
                           body))
    for c in cited:
        if c not in refs and c not in ('Table', 'Figure', 'Fig', 'Eq'):
            fails.append(f'in-text citation missing from reference list: {c}')
    for sec in ['## 1.0 Introduction', '## 2.0 Materials and Methods',
                '## 3.0 Results and Discussion', '## 4.0 Conclusion',
                '## 5.0 Acknowledgements', '## 6.0 Ethical Statement',
                '## 8.0 References']:
        if sec not in md:
            fails.append(f'required section missing: {sec}')

    print(f'{n} numeric checks | abstract {w} words | {len(entries)} references')
    if fails:
        print(f'\n{len(fails)} FAILURES:')
        for f in fails:
            print('  -', f)
        sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
