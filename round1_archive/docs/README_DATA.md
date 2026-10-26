# Adding Ethiopia, Tanzania and Uganda

## What to download

Three files. Right-click each link → **Save Link As**, or paste into your browser
address bar. Save all three into the **`Data`** folder next to `Kenya.tab`.

| Country | Spectra | Size | Download link |
|---|---|---|---|
| Ethiopia | 1,161 | 21 MB | https://data.worldagroforestry.org/api/access/datafile/10920 |
| Tanzania | ~2,100 | 37 MB | https://data.worldagroforestry.org/api/access/datafile/10931 |
| Uganda | ~570 | 10 MB | https://data.worldagroforestry.org/api/access/datafile/10932 |

**Rename each downloaded file** to exactly `Ethiopia.tab`, `Tanzania.tab` and
`Uganda.tab`. The browser may save them with the datafile number instead of a
name — the pipeline matches on filename, so the names must be exact.

You do **not** need to download anything else. The wet-chemistry deposit already
in `Data/African archived samples September 2018.tab` covers all 18 AfSIS
countries; only the spectral files are split per country.

## Verify before running

Check the sizes are right — a truncated download is the most common problem:

```
Ethiopia.tab   ~20.8 MB
Tanzania.tab   ~37.3 MB
Uganda.tab      ~9.9 MB
Kenya.tab      ~15.4 MB   (already present)
```

If a file is much smaller than listed, the download was interrupted. Delete and
retry.

## Then run

```bash
cd rebuild
python3 prep_ea.py          # assembles whatever country files are present
```

It prints a per-country summary. Expected once all four are in place:

| Country | Chemistry records | Approx. sites |
|---|---|---|
| Ethiopia | 381 | 5–6 |
| Tanzania | 367 | 5–6 |
| Kenya | 340 | 3 |
| Uganda | 222 | 3–4 |
| **Total** | **≈1,310** | **≈17–19** |

That takes the transfer analysis from 3 folds to roughly 17 leave-site-out
folds, which is the point of the exercise: the headline claim currently rests
on three comparisons, and with four countries it rests on seventeen.

## What the pipeline does automatically

`prep_ea.py` checks that every country file uses the same wavenumber grid and
stops with a clear message if not, so spectra are never silently concatenated
across incompatible instruments. It joins on sample number (SSN), reports any
attrition, and writes `ea/ea_audit.json` documenting sample counts, sites,
clusters, coordinate ranges and target completeness per country — which is the
data-provenance table the reviewer asked for.

## If a download fails

The ICRAF repository occasionally rate-limits. Wait a minute and retry, or go
via the dataset landing page and use the per-file Download button:

https://doi.org/10.34725/DVN/QXCWP1

Both deposits are open access and require no account.

## Citation

Cite both deposits in the manuscript (already in the reference list):

> Vågen, T.-G., Winowiecki, L. A., Desta, L., Tondoh, E. J., Weullow, E.,
> Shepherd, K. and Sila, A. (2020). *Mid-Infrared Spectra (MIRS) from ICRAF Soil
> and Plant Spectroscopy Laboratory: Africa Soil Information Service (AfSIS)
> Phase I 2009–2013*. World Agroforestry (ICRAF), V1.
> https://doi.org/10.34725/DVN/QXCWP1

> Vågen, T.-G., Winowiecki, L. A., Desta, L., Tondoh, J., Weullow, E.,
> Shepherd, K., Sila, A., Dunham, S. J., Hernández-Allica, J., Carter, J.,
> Durenkamp, M. and McGrath, S. P. (2021). *Wet chemistry data for a subset of
> AfSIS: Phase I archived soil samples*. World Agroforestry (ICRAF), V2.
> https://doi.org/10.34725/DVN/66BFOB
