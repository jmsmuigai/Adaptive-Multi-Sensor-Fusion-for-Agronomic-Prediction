# Attention-Weighted Hybrid CNN–RBF Model for MIR Soil Nutrient Prediction — Revision 2

**Comparison of a Hybrid Deep Learning Model with Machine Learning Methods for Mid-Infrared Soil Nutrient Prediction under Spatial Transfer in East Africa**

Running title: *Spectral soil nutrient prediction under spatial transfer*

Reproducibility package for the JAGST manuscript (ID 1283), revised in response to the reviewer's report (round 2, October 2026).

Mburu, J. M., Ogada, K. and Kimwele, M. — Department of Computing, Jomo Kenyatta University of Agriculture and Technology, Kenya.

---

## Summary of the revision

The round-2 analysis was rebuilt from scratch in response to the reviewer:

| Reviewer request | Where |
|---|---|
| Ablation of head, pooling, gate, ancillary input | `src/run_main.py --stage ablation`, Table 5, Fig. 6 |
| Direct boundedness test, ‖W_o‖₁, log vs original scale | `src/results.py: boundedness()`, Table 6 |
| 5 seeds for every neural model | `run_main.SEEDS` |
| Nested, group-aware tuning and early stopping | `common.inner_split`, `run_main.run_classical/run_neural` |
| Library baselines + agreement with round-1 NumPy code | `src/classical.py`, `src/agreement.py`, Table S4 |
| Cubist, MBL (local PLS), SVR, KRR, GPR, plain and multi-output 1D-CNN | `src/classical.py`, `src/nets.py` |
| Random, cluster, site, AEZ (Köppen–Geiger) and country designs | `common.outer_folds` |
| Group-bootstrap CIs, paired tests, Friedman/Nemenyi | `src/results.py` |
| RPD, RPIQ, CCC, RMSE/SD, MAE skill | `common.metrics`, `analyze.pooled_metrics` |
| Like-for-like Kenya test and learning curves (size vs diversity) | `src/exp_kenya.py`, `src/exp_lc.py` |
| Local spiking k = 0, 5, 10, 20, 50 | `src/exp_spike.py` |
| Group-aware conformal prediction | `src/exp_conformal.py` |
| Attention vs occlusion / integrated gradients / PLSR VIP + permutation test | `src/exp_interp.py` |
| Preprocessing sensitivity, CO₂ band masked | `src/exp_prep.py` |

### Headline results (pooled out-of-fold R², mean of 13 properties)

| Design | Median model | GPR (best mean rank) | Hybrid |
|---|---|---|---|
| Random | 0.744 | 0.789 | 0.654 |
| Leave-cluster-out | 0.711 | 0.797 | 0.606 |
| Leave-site-out | 0.507 | 0.545 | 0.411 |
| Leave-AEZ-out | 0.453 | 0.509 | 0.359 |
| Leave-country-out | 0.436 | 0.467 | 0.290 |

The hybrid ranked last in every design. In the ablations, the RBF head and the modality gate added no skill, while positional information and position-aware pooling were essential. Calibration diversity mattered more than size, and group-aware conformal intervals improved coverage under transfer. Olsen P was not predictable in unsampled landscapes.

### Corrections to round 1 (disclosed in the response letter)

- The round-1 MLP and hybrid early-stopped on the **test** fold. This is fixed: early stopping now uses a group-disjoint inner split.
- The round-1 NumPy PLSR computed its coefficients without the weight matrix. It is replaced by scikit-learn.
- Round-1 hyperparameters were chosen on one reserved fold. They are now selected by nested, group-aware tuning.

The round-1 code and outputs are kept unchanged in `round1_archive/` for transparency. **Do not use them for results.**

## Layout

```
src/            round-2 pipeline (data, models, designs, experiments, tables, figures, verification)
src/ms/         manuscript sources (Markdown) and the Word build scripts
results/        tables (CSV), numbers.json (every number quoted in the text), agreement.csv,
                records_main_and_ablation.tar.xz  (5,495 per-fit JSON records)
                records_experiments.tar.xz        (spiking, learning curves, Kenya, conformal, interpretability, preprocessing)
figures/        Figures 1–10 (300 dpi)
docs/           revised manuscript, response to the reviewer, supplementary material (Markdown renderings)
round1_archive/ superseded round-1 code and outputs
```

## Reproduce

1. Download the two AfSIS deposits (`docs/README_DATA.md`) and run `round1_archive/src/prep_ea.py`. Copy the outputs `ea_spectra.npz`, `ea_meta.csv` and `ea_targets.csv` into `src/data/`.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Run the experiments. Every runner is resumable and skips finished records:
   ```
   cd src
   python3 run_main.py --stage main --workers 4
   python3 run_main.py --stage ablation --workers 4
   for e in agreement exp_prep exp_lc exp_spike exp_kenya exp_conformal exp_interp; do python3 jobrun.py $e 4; done
   ```
   About 85,000 model fits in total. To skip this step, unpack the two archives in `results/` into `src/out/` instead.
4. Rebuild every number, table, figure and Word document: `bash make_docs.sh`. The four JAGST submission files (Calibri 11 pt, single spacing, light figures) are built with `cd src/ms && FIGDIR=fig_final python3 make_final.py`. `verify.py` recomputes the headline numbers independently from the raw records.

Seeds: fold assignment 20260804; neural seeds 11, 22, 33, 44, 55.

## Data

- MIR spectra: Vågen *et al.* (2020), doi:10.34725/DVN/QXCWP1
- Wet chemistry: Vågen *et al.* (2021), doi:10.34725/DVN/66BFOB

## Licence

MIT (see `LICENSE`). The data remain under their original licences.
