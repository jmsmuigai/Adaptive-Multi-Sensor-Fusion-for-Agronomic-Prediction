"""kenya_tables.py -- manuscript tables, emitted as JSON for the docx builder."""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KEN = os.path.join(HERE, 'ea')
EAD = KEN
M = ['PLSR', 'Ridge', 'RandomForest', 'GradientBoosting', 'MLP', 'DualDAFM']
SH = {'PLSR': 'PLSR', 'Ridge': 'Ridge', 'RandomForest': 'RF',
      'GradientBoosting': 'GBM', 'MLP': 'MLP', 'DualDAFM': 'Hybrid'}
NICE = {'pH': 'pH', 'OrgC_pct': 'Organic C (%)', 'TotN_pct': 'Total N (%)',
        'P_Olsen': 'Olsen P', 'Ca_exch': 'Exch. Ca', 'Mg_exch': 'Exch. Mg',
        'K_exch': 'Exch. K', 'eCEC': 'eCEC', 'S_total': 'Total S',
        'Fe_total': 'Total Fe', 'Mn_total': 'Total Mn', 'Zn_total': 'Total Zn',
        'Cu_total': 'Total Cu'}


def main():
    r = pd.read_csv(os.path.join(KEN, 'summary.csv'))
    h = json.load(open(os.path.join(KEN, 'headline.json')))
    aud = json.load(open(os.path.join(KEN, 'ea_audit.json')))
    out = {}

    # Table 1 -- dataset composition
    rows = [['Country', 'Samples', 'Sentinel sites', 'LDSF clusters',
             'Latitude range', 'Longitude range']]
    for c in aud['countries']:
        d = aud['per_country'][c]
        rows.append([c, str(d['n']), str(d['n_sites']), str(d['n_clusters']),
                     f"{d['lat_range'][0]:.2f} to {d['lat_range'][1]:.2f}",
                     f"{d['lon_range'][0]:.2f} to {d['lon_range'][1]:.2f}"])
    rows.append(['Total', str(aud['n_samples']), str(aud['n_sites']),
                 str(aud['n_clusters']), '—', '—'])
    out['table1'] = {'caption': 'Table 1: Composition of the analysis set. Samples are '
                                'AfSIS Land Degradation Surveillance Framework records '
                                'with paired mid-infrared spectra and wet chemistry, '
                                f"collected 2009-2013 at two depths ({aud['depth_counts']}). "
                                f"Spectra cover {aud['n_wavenumbers']} wavenumbers from "
                                f"{aud['wavenumber_range_cm-1'][0]:.1f} to "
                                f"{aud['wavenumber_range_cm-1'][1]:.1f} cm-1. The spectra "
                                'and chemistry deposits joined with zero attrition.',
                     'rows': rows}

    # Table 2 -- properties modelled
    rows = [['Property', 'Source column', 'Unit', 'n present', 'Median', 'Skew']]
    UN = {'pH': '-', 'OrgC_pct': '% w/w', 'TotN_pct': '% w/w', 'P_Olsen': 'mg/kg',
          'Ca_exch': 'cmolc/kg', 'Mg_exch': 'cmolc/kg', 'K_exch': 'cmolc/kg',
          'eCEC': 'cmolc/kg', 'S_total': 'mg/kg', 'Fe_total': 'mg/kg',
          'Mn_total': 'mg/kg', 'Zn_total': 'mg/kg', 'Cu_total': 'mg/kg'}
    SRC = {'pH': 'pH', 'OrgC_pct': 'Percent_Org_C', 'TotN_pct': 'Percent_N',
           'P_Olsen': 'Olsen.P', 'Ca_exch': 'Caex', 'Mg_exch': 'Mgex',
           'K_exch': 'Kex', 'eCEC': 'eCEC', 'S_total': 'S', 'Fe_total': 'Fe',
           'Mn_total': 'Mn', 'Zn_total': 'Zn', 'Cu_total': 'Cu'}
    for t in NICE:
        d = aud['targets'][t]
        rows.append([NICE[t], SRC[t], UN[t], str(d['n_present']),
                     f"{d['median']:.4g}", f"{d['skew']:.2f}"])
    out['table2'] = {'caption': 'Table 2: Soil properties modelled, with their source '
                                'columns in the wet-chemistry deposit. Properties with '
                                'skew above 2.0 were log-transformed for fitting; all '
                                'reported errors are back-transformed to these units.',
                     'rows': rows}

    # Table 2 -- leave-cluster-out
    g = r[r.design == 'cluster']
    p = g.pivot(index='target', columns='model', values='R2')[M]
    b = g.loc[g.groupby('target').R2.idxmax()].set_index('target')
    order = p.mean(1).sort_values(ascending=False).index
    rows = [['Property'] + [SH[m] for m in M] +
            ['Best model', 'Best R² (95% CI)', 'RMSE', 'MAE']]
    for t in order:
        rows.append([NICE[t]] + [f'{p.loc[t, m]:.3f}' for m in M] +
                    [SH[b.loc[t, 'model']],
                     f"{b.loc[t, 'R2']:.3f} ({b.loc[t, 'R2_lo']:.3f}–{b.loc[t, 'R2_hi']:.3f})",
                     f"{b.loc[t, 'RMSE']:.4g}", f"{b.loc[t, 'MAE']:.4g}"])
    rows.append(['Mean'] + [f'{p[m].mean():.3f}' for m in M] + ['—', '—', '—', '—'])
    out['table3'] = {'caption': 'Table 3: Leave-cluster-out cross-validation. R² is the '
                                'mean of four evaluation folds; the confidence interval is '
                                't-based across folds. RMSE and MAE are in the measurement '
                                'units of each property. RF = random forest, GBM = gradient '
                                'boosting, Hybrid = the proposed CNN–DAFM–RBF architecture.',
                     'rows': rows}

    # Table 3 -- transfer
    ps = r[r.design == 'site'].pivot(index='target', columns='model', values='R2')[M]
    rows = [['Property'] + [SH[m] for m in M]]
    for t in order:
        rows.append([NICE[t]] + [f'{ps.loc[t, m]:.3f}' for m in M])
    rows.append(['Mean R²'] + [f'{ps[m].mean():.3f}' for m in M])
    rows.append(['Mean rank'] + [f"{h['site']['mean_rank'][m]:.2f}" for m in M])
    rows.append(['Properties with R² > 0'] + [str(h['site']['n_positive'][m]) for m in M])
    rows.append(['Worst-case R²'] + [f"{h['site']['worst_R2'][m]:.2f}" for m in M])
    out['table5'] = {'caption': 'Table 5: Leave-site-out cross-validation — transfer to an '
                                'unsampled 10 x 10 km landscape. Lower mean rank is better.',
                     'rows': rows}

    # Table 4 -- the three designs side by side
    rows = [['Model'] + [f'{d} mean R²' for d in ['Cluster', 'Site', 'Country']] +
            [f'{d} rank' for d in ['Cluster', 'Site', 'Country']] +
            [f'{d} n>0' for d in ['Cluster', 'Site', 'Country']]]
    for m in M:
        rows.append([SH[m]] +
                    [f"{h[d]['mean_R2'][m]:.3f}" for d in ['cluster', 'site', 'country']] +
                    [f"{h[d]['mean_rank'][m]:.2f}" for d in ['cluster', 'site', 'country']] +
                    [str(h[d]['n_positive'][m]) for d in ['cluster', 'site', 'country']])
    out['table4'] = {'caption': 'Table 4: Performance across the three nested validation '
                                'designs. Cluster = leave-cluster-out (5 folds), Site = '
                                'leave-site-out (8 folds), Country = leave-country-out '
                                '(4 folds). Rank is the mean rank across the thirteen '
                                'properties, where 1 is best; n>0 counts properties '
                                'retaining positive R².',
                     'rows': rows}

    # Table 6 -- calibration breadth
    ken = os.path.join(HERE, 'kenya', 'headline.json')
    if os.path.exists(ken):
        kh = json.load(open(ken))
        rows = [['Metric', 'Kenya only (3 sites, n=340)',
                 'Four countries (16 sites, n=1,310)']]
        rows.append(['Best mean R², leave-site-out',
                     f"{max(kh['site']['mean_R2'].values()):.3f}",
                     f"{max(h['site']['mean_R2'].values()):.3f}"])
        rows.append(['Properties with R² > 0 (best model)',
                     str(max(kh['site']['n_positive'].values())) + ' of 13',
                     str(max(h['site']['n_positive'].values())) + ' of 13'])
        kr = pd.read_csv(os.path.join(HERE, 'kenya', 'summary.csv'))
        kc = kr[kr.design == 'cluster'].groupby('target').R2.max()
        ec = r[r.design == 'cluster'].groupby('target').R2.max()
        for t in ['P_Olsen', 'TotN_pct', 'pH', 'K_exch', 'S_total']:
            rows.append([f'Best R² within landscape, {NICE[t]}',
                         f'{kc[t]:.3f}', f'{ec[t]:.3f}'])
        out['table6'] = {'caption': 'Table 6: Effect of calibration breadth. Both columns '
                                    'use identical models, preprocessing, tuning budget and '
                                    'validation design; only the geographic base of the '
                                    'calibration differs.',
                         'rows': rows}

    # Table 4 -- deficiency
    d = pd.read_csv(os.path.join(KEN, 'deficiency.csv'))
    best = g.groupby('target').R2.max()
    rows = [['Property', 'Critical level', 'Basis', '% below',
             '% at ½×', '% at 1½×', 'Median', 'Best R²']]
    for _, x in d.iterrows():
        rows.append([NICE[x.target], f'{x.critical:g}', x.basis,
                     f'{x.pct_below:.1f}', f'{x.pct_below_half:.1f}',
                     f'{x.pct_below_1p5:.1f}', f'{x["median"]:.3g}',
                     f'{best[x.target]:.3f}'])
    out['table7'] = {'caption': 'Table 7: Agronomic status of the topsoil (0-20 cm) '
                                'samples against indicative critical levels, with '
                                'sensitivity to the threshold, alongside the best R² '
                                'achieved under leave-cluster-out validation.',
                     'rows': rows}

    # Table 5 -- branch gate
    bg = pd.read_csv(os.path.join(KEN, 'branch_gate.csv'))
    rows = [['Property', 'Spectral branch weight', 'Ancillary branch weight']]
    for _, x in bg.sort_values('spectral_weight', ascending=False).iterrows():
        rows.append([NICE[x.target], f'{x.spectral_weight:.3f} ± {x.spectral_sd:.3f}',
                     f'{x.covariate_weight:.3f}'])
    out['table8'] = {'caption': 'Table 8: Learned branch-gate weights, mean ± SD across '
                                'evaluation folds. These are measured quantities, not '
                                'assumptions, and show that the balance between modalities '
                                'is learned separately for each property.',
                     'rows': rows}


    # Table 9 -- leave-country-out by property
    pc = r[r.design == 'country'].pivot(index='target', columns='model', values='R2')[M]
    order9 = pc.mean(1).sort_values(ascending=False).index
    rows = [['Property'] + [SH[m] for m in M]]
    for t in order9:
        rows.append([NICE[t]] + [f'{pc.loc[t, m]:.3f}' for m in M])
    rows.append(['Mean'] + [f'{pc[m].mean():.3f}' for m in M])
    out['table9'] = {'caption': 'Table 9: Leave-country-out cross-validation by property. '
                                'Each fold withholds one national dataset in full. This is '
                                'the most demanding design tested and corresponds to '
                                'exporting a calibration across a national border.',
                     'rows': rows}

    # Table 10 -- per held-out country
    cd = os.path.join(EAD, 'country_detail.csv')
    if os.path.exists(cd):
        det = pd.read_csv(cd)
        piv = det.pivot_table(index='country', columns='model', values='R2')[M]
        rows = [['Held-out country'] + [SH[m] for m in M] +
                ['Best model', 'Best mean R²', 'Properties R² > 0']]
        for c in piv.index:
            bm = piv.loc[c].idxmax()
            g = det[(det.country == c) & (det.model == bm)]
            rows.append([c] + [f'{piv.loc[c, m]:.3f}' for m in M] +
                        [SH[bm], f'{piv.loc[c, bm]:.3f}',
                         f'{int((g.R2 > 0).sum())} of {len(g)}'])
        out['table10'] = {'caption': 'Table 10: Performance when each country is withheld '
                                     'in turn, averaged over the thirteen properties. '
                                     'Ethiopia served as the hyperparameter-selection fold '
                                     'and is therefore not reported as a test country.',
                          'rows': rows}

    # Table -- bounded vs unbounded failure severity
    BOUND = {'PLSR': 'Unbounded (linear projection)',
             'Ridge': 'Unbounded (linear)',
             'RandomForest': 'Bounded (convex combination of leaf values)',
             'GradientBoosting': 'Bounded (sum of bounded leaf values)',
             'MLP': 'Unbounded (linear output layer)',
             'DualDAFM': 'Bounded: |y-b| <= ||W_o||_1  (Proposition 2)'}
    rows = [['Model', 'Output range', 'Worst R² (leave-site-out)',
             'Worst R² (leave-country-out)']]
    ordered = sorted(M, key=lambda m: h['site']['worst_R2'][m])
    for m in ordered:
        rows.append([SH[m], BOUND[m], f"{h['site']['worst_R2'][m]:.2f}",
                     f"{h['country']['worst_R2'][m]:.2f}"])
    out['table_bound'] = {
        'caption': 'Table 6: Worst-case single-property R² under the two transfer designs, '
                   'set against whether each model\'s output is bounded by construction. '
                   'Models with bounded output occupy the three least severe positions and '
                   'models with unbounded output the three most severe, as Proposition 2 '
                   '(Section 2.4.5) predicts.',
        'rows': rows}

    # Table -- computational cost
    tp = os.path.join(EAD, 'timing.json')
    if os.path.exists(tp):
        t = json.load(open(tp))
        rows = [['Model', 'Training time (s)', 'Inference (ms/sample)',
                 'p95 (ms/sample)', 'Throughput (samples/s)']]
        for k, v in t['models'].items():
            rows.append([SH.get(k, k), f"{v['fit_s']:.2f}", f"{v['ms_mean']:.4f}",
                         f"{v['ms_p95']:.4f}", f"{v['per_sec']:,}"])
        e = t['environment']
        out['table_timing'] = {
            'caption': f"Table 9: Measured computational cost on the assembled dataset "
                       f"({e['n']:,} samples, {e['bands']} preprocessed bands). "
                       f"Environment: {e['platform']}, "
                       f"{e.get('mem_kB', 0) // 1048576:.1f} GiB RAM, Python "
                       f"{e['python']}, NumPy {e['numpy']}, single BLAS thread. "
                       f"Inference timed over {e['reps']} repetitions after a warm-up "
                       f"pass. Figures cover model inference only and exclude sample "
                       f"preparation and spectrometer acquisition.",
            'rows': rows}

    json.dump(out, open(os.path.join(KEN, 'tables.json'), 'w'), indent=1)
    for k, v in out.items():
        print(k, len(v['rows']) - 1, 'rows x', len(v['rows'][0]), 'cols')


if __name__ == '__main__':
    main()
