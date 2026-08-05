"""
prep_ea.py -- Assemble the East African analysis set from the AfSIS deposits.

Generalises prep_kenya.py to any subset of the per-country spectral files. The
wet-chemistry deposit already covers all countries, so only the spectral files
differ. Countries present are detected automatically from what has been
downloaded into ../Data, so the script runs whether one file or four are there.

Expected spectral files in ../Data (download links in README_DATA.md):
    Kenya.tab      862 spectra   (AfSIS wet chemistry: 340 Kenyan records)
    Ethiopia.tab  1161 spectra   (                      381 records)
    Tanzania.tab  ~2100 spectra  (                      367 records)
    Uganda.tab     ~570 spectra  (                      222 records)

Grouping levels available for validation, from tightest to loosest:
    Cluster  -> LDSF cluster within a sentinel site
    Site     -> LDSF sentinel site (a 10 x 10 km block)
    Country  -> national boundary
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'Data')
OUT = os.path.join(HERE, 'ea')
os.makedirs(OUT, exist_ok=True)

CANDIDATES = ['Kenya', 'Ethiopia', 'Tanzania', 'Uganda']

TARGETS = {
    'pH': 'pH', 'OrgC_pct': 'Percent_Org_C', 'TotN_pct': 'Percent_N',
    'P_Olsen': 'Olsen.P', 'Ca_exch': 'Caex', 'Mg_exch': 'Mgex',
    'K_exch': 'Kex', 'eCEC': 'eCEC', 'S_total': 'S', 'Fe_total': 'Fe',
    'Mn_total': 'Mn', 'Zn_total': 'Zn', 'Cu_total': 'Cu',
}
UNITS = {'pH': '-', 'OrgC_pct': '%', 'TotN_pct': '%', 'P_Olsen': 'mg/kg',
         'Ca_exch': 'cmolc/kg', 'Mg_exch': 'cmolc/kg', 'K_exch': 'cmolc/kg',
         'eCEC': 'cmolc/kg', 'S_total': 'mg/kg', 'Fe_total': 'mg/kg',
         'Mn_total': 'mg/kg', 'Zn_total': 'mg/kg', 'Cu_total': 'mg/kg'}


def main():
    chem = pd.read_csv(
        os.path.join(DATA, 'African archived samples September 2018.tab'), sep='\t')

    frames, present, wn_ref = [], [], None
    for c in CANDIDATES:
        p = os.path.join(DATA, f'{c}.tab')
        if not os.path.exists(p):
            print(f'  [skip] {c}.tab not found')
            continue
        s = pd.read_csv(p, sep='\t')
        wn_cols = [x for x in s.columns if x.startswith('m')]
        wn = np.array([float(x[1:]) for x in wn_cols])
        o = np.argsort(wn)
        wn_cols = [wn_cols[i] for i in o]
        wn = wn[o]
        if wn_ref is None:
            wn_ref, wn_ref_cols = wn, wn_cols
        else:
            if len(wn) != len(wn_ref) or not np.allclose(wn, wn_ref):
                raise SystemExit(f'{c}: wavenumber grid differs from Kenya; '
                                 'harmonisation would be required')
        frames.append(s[['SSN'] + wn_ref_cols])
        present.append(c)
        print(f'  [ok]   {c}.tab  {len(s)} spectra')

    if not frames:
        raise SystemExit('No spectral files found in ../Data')

    spec = pd.concat(frames, ignore_index=True).drop_duplicates('SSN')
    sub = chem[chem.Country.isin(present)].copy()
    df = sub.merge(spec, on='SSN', how='inner', validate='one_to_one')

    meta = df[['SSN', 'Country', 'Site', 'Cluster', 'Plot',
               'Latitude', 'Longitude', 'Depth']].copy()
    meta['SiteKey'] = meta.Country.astype(str) + '_' + meta.Site.astype(str)
    meta['ClusterKey'] = meta.SiteKey + '_' + meta.Cluster.astype(str)
    meta['is_topsoil'] = (meta.Depth == 'Topsoil').astype(int)

    y = pd.DataFrame({k: df[v] for k, v in TARGETS.items()})
    X = df[wn_ref_cols].to_numpy(dtype=float)

    np.savez_compressed(os.path.join(OUT, 'ea_spectra.npz'),
                        X=X, wavenumbers=wn_ref)
    meta.to_csv(os.path.join(OUT, 'ea_meta.csv'), index=False)
    y.to_csv(os.path.join(OUT, 'ea_targets.csv'), index=False)

    audit = {
        'countries': present,
        'n_samples': int(len(df)),
        'n_wavenumbers': int(len(wn_ref)),
        'wavenumber_range_cm-1': [float(wn_ref.min()), float(wn_ref.max())],
        'n_sites': int(meta.SiteKey.nunique()),
        'n_clusters': int(meta.ClusterKey.nunique()),
        'per_country': {c: {
            'n': int((meta.Country == c).sum()),
            'n_sites': int(meta[meta.Country == c].SiteKey.nunique()),
            'n_clusters': int(meta[meta.Country == c].ClusterKey.nunique()),
            'lat_range': [round(float(meta[meta.Country == c].Latitude.min()), 3),
                          round(float(meta[meta.Country == c].Latitude.max()), 3)],
            'lon_range': [round(float(meta[meta.Country == c].Longitude.min()), 3),
                          round(float(meta[meta.Country == c].Longitude.max()), 3)],
        } for c in present},
        'sites': {s: {'country': str(meta[meta.SiteKey == s].Country.iloc[0]),
                      'n': int((meta.SiteKey == s).sum()),
                      'n_clusters': int(meta[meta.SiteKey == s].ClusterKey.nunique()),
                      'lat': round(float(meta[meta.SiteKey == s].Latitude.mean()), 4),
                      'lon': round(float(meta[meta.SiteKey == s].Longitude.mean()), 4)}
                  for s in sorted(meta.SiteKey.unique())},
        'depth_counts': meta.Depth.value_counts().to_dict(),
        'targets': {k: {'unit': UNITS[k],
                        'n_present': int(y[k].notna().sum()),
                        'median': round(float(y[k].median()), 4),
                        'skew': round(float(y[k].skew()), 3)} for k in TARGETS},
        'join': {'chem_rows_for_countries': int(len(sub)),
                 'joined': int(len(df)),
                 'attrition': int(len(sub) - len(df))},
        'sources': {'spectra': 'doi:10.34725/DVN/QXCWP1',
                    'chemistry': 'doi:10.34725/DVN/66BFOB'},
    }
    json.dump(audit, open(os.path.join(OUT, 'ea_audit.json'), 'w'), indent=1)
    print(f"\nAssembled {audit['n_samples']} samples | "
          f"{audit['n_sites']} sites | {audit['n_clusters']} clusters | "
          f"countries: {', '.join(present)}")
    for c, d in audit['per_country'].items():
        print(f"   {c:10s} n={d['n']:4d}  sites={d['n_sites']:2d}  "
              f"clusters={d['n_clusters']:3d}")
    if audit['join']['attrition']:
        print(f"   NOTE: {audit['join']['attrition']} chemistry records had no "
              f"matching spectrum (missing country file?)")


if __name__ == '__main__':
    main()
