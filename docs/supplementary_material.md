**Supplementary Material**

**Does Architectural Complexity Pay Off Under Spatial Transfer? Ablation, Uncertainty and Nested Validation of an Attention-Weighted Hybrid CNN–RBF Model for Mid-Infrared Soil Nutrient Prediction in East Africa**

Table S1: Composition of the outer folds of each validation design. Site names are followed by country and sample count.

| Design | Fold | Test n | Withheld units |
| --- | ---: | ---: | ---: |
| Leave-cluster-out | 1 | 262 | 41 clusters from 14 sites |
| Leave-cluster-out | 2 | 262 | 41 clusters from 14 sites |
| Leave-cluster-out | 3 | 262 | 41 clusters from 14 sites |
| Leave-cluster-out | 4 | 262 | 42 clusters from 14 sites |
| Leave-cluster-out | 5 | 262 | 42 clusters from 14 sites |
| Leave-site-out | 1 | 156 | Kidatu (Tanzania, n=156) |
| Leave-site-out | 2 | 126 | Mbinga (Tanzania, n=1); Namasuba (Uganda, n=125) |
| Leave-site-out | 3 | 126 | Dambidolo (Ethiopia, n=121); Bukwaya (Tanzania, n=2); Kiberashi (Tanzania, n=3) |
| Leave-site-out | 4 | 144 | Mega (Ethiopia, n=120); Pandambili (Tanzania, n=24) |
| Leave-site-out | 5 | 178 | Merar (Ethiopia, n=59); Mbalambala (Kenya, n=119) |
| Leave-site-out | 6 | 181 | Chinyanghuku (Tanzania, n=69); Itende (Tanzania, n=112) |
| Leave-site-out | 7 | 192 | Kutaber (Ethiopia, n=81); Marafa (Kenya, n=111) |
| Leave-site-out | 8 | 207 | Morijo (Kenya, n=110); Hoima (Uganda, n=97) |
| Leave-AEZ-out | 1 | 379 | Tropical savanna (Aw): Dambidolo, Bukwaya, Kiberashi, Kidatu, Hoima |
| Leave-AEZ-out | 2 | 371 | Temperate highland (Cfb/Cwb/Csb/Cwa): Kutaber, Mega, Merar, Morijo, Mbinga |
| Leave-AEZ-out | 3 | 316 | Hot semi-arid (BSh): Marafa, Chinyanghuku, Itende, Pandambili |
| Leave-AEZ-out | 4 | 125 | Tropical monsoon (Am): Namasuba |
| Leave-AEZ-out | 5 | 119 | Hot arid (BWh): Mbalambala |
| Leave-country-out | 1 | 381 | Ethiopia |
| Leave-country-out | 2 | 367 | Tanzania |
| Leave-country-out | 3 | 340 | Kenya |
| Leave-country-out | 4 | 222 | Uganda |
| Random | 1–5 | 262 each | random partition (seed 20260804) |


Table S2: Hyperparameter search spaces. Selection always used training-fold data only, with a group-disjoint inner split or inner folds (Section 2.6). Neural configurations list learning rate (lr), convolution channels (c1, c2), first kernel size (k1) and stride (s1), hidden widths (h, h1, h2), RBF units (R), dropout and weight decay (wd); unspecified values are as in BASE: c1 = 16, c2 = 32, k1 = 11, s1 = 2, k2 = 7, attention dimension 32, ancillary embedding 8, RBF projection 8.

| Model | Selection | Candidate configurations |
| --- | ---: | ---: |
| PLSR | 3 group-disjoint inner folds | n_components=4; n_components=8; n_components=12; n_components=16; n_components=20; n_components=25; n_components=30 |
| Ridge | 3 group-disjoint inner folds | alpha=0.1; alpha=0.316; alpha=1; alpha=3.16; alpha=10; alpha=31.6; alpha=100; alpha=316; alpha=1e+03; alpha=3.16e+03; alpha=1e+04 |
| SVR | 3 group-disjoint inner folds | C=1, gamma=0.000706, epsilon=0.1; C=1, gamma=0.00235, epsilon=0.1; C=1, gamma=0.00706, epsilon=0.1; C=10, gamma=0.000706, epsilon=0.1; C=10, gamma=0.00235, epsilon=0.1; C=10, gamma=0.00706, epsilon=0.1; C=100, gamma=0.000706, epsilon=0.1; C=100, gamma=0.00235, epsilon=0.1; C=100, gamma=0.00706, epsilon=0.1 |
| KRR | 3 group-disjoint inner folds | alpha=0.01, gamma=0.000706; alpha=0.01, gamma=0.00235; alpha=0.01, gamma=0.00706; alpha=0.1, gamma=0.000706; alpha=0.1, gamma=0.00235; alpha=0.1, gamma=0.00706; alpha=1, gamma=0.000706; alpha=1, gamma=0.00235; alpha=1, gamma=0.00706 |
| GPR | none (fixed settings; GPR kernel by marginal likelihood) | – |
| Cubist | one group-disjoint inner split | n_committees=5, neighbors=None; n_committees=5, neighbors=5; n_committees=20, neighbors=None; n_committees=20, neighbors=5 |
| MBL | one group-disjoint inner split | k=75; k=150; k=300 |
| RF | none (fixed settings; GPR kernel by marginal likelihood) | max_features=0.33 |
| LightGBM | one group-disjoint inner split | num_leaves=7; num_leaves=31 |
| MLP | one group-disjoint inner split; early stopping on the same split | {lr=0.001, h1=128, h2=64, dropout=0.1}; {lr=0.001, h1=256, h2=128, dropout=0.3}; {lr=0.003, h1=64, h2=32, dropout=0.0, wd=0.001} |
| CNN | one group-disjoint inner split; early stopping on the same split | {lr=0.003, c1=16, c2=32, k1=11, s1=2, h=64, dropout=0.2}; {lr=0.001, c1=32, c2=64, k1=11, s1=2, h=64, dropout=0.2}; {lr=0.003, c1=16, c2=32, k1=15, s1=3, h=128, dropout=0.3, wd=0.001} |
| MOCNN | one group-disjoint inner split; early stopping on the same split | {lr=0.003, c1=16, c2=32, k1=11, s1=2, h=64, dropout=0.2}; {lr=0.001, c1=32, c2=64, k1=11, s1=2, h=64, dropout=0.2}; {lr=0.003, c1=16, c2=32, k1=15, s1=3, h=128, dropout=0.3, wd=0.001} |
| Hybrid | one group-disjoint inner split; early stopping on the same split | {lr=0.003, c1=16, c2=32, k1=11, s1=2, R=16, dropout=0.0, wd=0.0001}; {lr=0.003, c1=32, c2=64, k1=11, s1=2, R=32, dropout=0.0, wd=0.0001}; {lr=0.003, c1=16, c2=32, k1=15, s1=3, R=16, dropout=0.1, wd=0.001} |


Table S3: Pooled out-of-fold R² for every model, property and validation design (seed mean for neural models; predictions limited to the training range). RMSE, RPD, RPIQ, CCC, MAE skill and log-scale R² for every combination are provided in the released results file (pooled_metrics_all.csv).

| Design | Property | PLSR | Ridge | SVR | KRR | GPR | Cubist | MBL | RF | LightGBM | MLP | CNN | MO-CNN | Hybrid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | pH | 0.90 | 0.91 | 0.94 | 0.94 | 0.94 | 0.91 | 0.92 | 0.88 | 0.91 | 0.92 | – | – | 0.88 |
| random | Organic C | 0.89 | 0.89 | 0.91 | 0.89 | 0.89 | 0.89 | 0.88 | 0.84 | 0.86 | 0.88 | – | – | 0.81 |
| random | Total N | 0.92 | 0.92 | 0.94 | 0.93 | 0.93 | 0.91 | 0.93 | 0.87 | 0.88 | 0.92 | – | – | 0.84 |
| random | Olsen P | 0.40 | 0.40 | 0.52 | 0.40 | 0.36 | 0.19 | 0.37 | 0.12 | 0.28 | 0.38 | – | – | 0.11 |
| random | Exch. Ca | 0.90 | 0.90 | 0.92 | 0.93 | 0.93 | 0.90 | 0.91 | 0.88 | 0.90 | 0.91 | – | – | 0.88 |
| random | Exch. Mg | 0.79 | 0.83 | 0.88 | 0.87 | 0.87 | 0.83 | 0.85 | 0.81 | 0.82 | 0.83 | – | – | 0.78 |
| random | Exch. K | 0.42 | 0.47 | 0.52 | 0.51 | 0.50 | 0.33 | 0.40 | 0.29 | 0.40 | 0.37 | – | – | 0.20 |
| random | eCEC | 0.91 | 0.92 | 0.93 | 0.93 | 0.93 | 0.91 | 0.92 | 0.91 | 0.92 | 0.92 | – | – | 0.90 |
| random | Total S | 0.53 | 0.46 | 0.66 | 0.67 | 0.66 | 0.41 | 0.64 | 0.29 | 0.42 | 0.52 | – | – | 0.28 |
| random | Total Fe | 0.90 | 0.91 | 0.93 | 0.93 | 0.93 | 0.93 | 0.92 | 0.88 | 0.92 | 0.91 | – | – | 0.88 |
| random | Total Mn | 0.49 | 0.57 | 0.69 | 0.66 | 0.67 | 0.63 | 0.62 | 0.62 | 0.66 | 0.63 | – | – | 0.57 |
| random | Total Zn | 0.77 | 0.79 | 0.88 | 0.84 | 0.83 | 0.84 | 0.85 | 0.82 | 0.84 | 0.84 | – | – | 0.78 |
| random | Total Cu | 0.68 | 0.70 | 0.80 | 0.81 | 0.80 | 0.68 | 0.78 | 0.62 | 0.69 | 0.74 | – | – | 0.60 |
| cluster | pH | 0.88 | 0.90 | 0.92 | 0.93 | 0.93 | 0.87 | 0.89 | 0.87 | 0.89 | 0.91 | 0.91 | 0.87 | 0.87 |
| cluster | Organic C | 0.91 | 0.92 | 0.91 | 0.90 | 0.91 | 0.91 | 0.92 | 0.86 | 0.87 | 0.87 | 0.87 | 0.86 | 0.76 |
| cluster | Total N | 0.92 | 0.92 | 0.93 | 0.93 | 0.93 | 0.92 | 0.93 | 0.88 | 0.89 | 0.92 | 0.91 | 0.90 | 0.85 |
| cluster | Olsen P | 0.38 | 0.35 | 0.46 | 0.45 | 0.47 | 0.38 | 0.42 | 0.23 | 0.43 | 0.36 | 0.36 | 0.36 | 0.09 |
| cluster | Exch. Ca | 0.88 | 0.89 | 0.91 | 0.92 | 0.92 | 0.88 | 0.89 | 0.86 | 0.88 | 0.89 | 0.89 | 0.88 | 0.84 |
| cluster | Exch. Mg | 0.74 | 0.79 | 0.81 | 0.83 | 0.83 | 0.78 | 0.80 | 0.78 | 0.80 | 0.80 | 0.78 | 0.78 | 0.73 |
| cluster | Exch. K | 0.39 | 0.41 | 0.48 | 0.47 | 0.45 | 0.34 | 0.39 | 0.31 | 0.39 | 0.27 | 0.27 | 0.32 | 0.19 |
| cluster | eCEC | 0.90 | 0.91 | 0.92 | 0.92 | 0.92 | 0.90 | 0.91 | 0.89 | 0.91 | 0.91 | 0.91 | 0.90 | 0.89 |
| cluster | Total S | 0.23 | 0.69 | 0.83 | 0.83 | 0.82 | 0.22 | 0.26 | 0.23 | 0.32 | 0.38 | 0.22 | 0.26 | 0.18 |
| cluster | Total Fe | 0.89 | 0.90 | 0.93 | 0.93 | 0.93 | 0.87 | 0.92 | 0.86 | 0.90 | 0.91 | 0.90 | 0.83 | 0.83 |
| cluster | Total Mn | 0.42 | 0.41 | 0.64 | 0.66 | 0.67 | 0.59 | 0.54 | 0.56 | 0.61 | 0.61 | 0.59 | 0.59 | 0.46 |
| cluster | Total Zn | 0.71 | 0.73 | 0.83 | 0.83 | 0.82 | 0.79 | 0.82 | 0.78 | 0.79 | 0.78 | 0.79 | 0.76 | 0.70 |
| cluster | Total Cu | 0.64 | 0.66 | 0.74 | 0.72 | 0.75 | 0.63 | 0.72 | 0.59 | 0.64 | 0.63 | 0.61 | 0.57 | 0.50 |
| site | pH | 0.70 | 0.75 | 0.86 | 0.85 | 0.85 | 0.79 | 0.74 | 0.76 | 0.81 | 0.85 | 0.82 | 0.76 | 0.69 |
| site | Organic C | 0.86 | 0.89 | 0.85 | 0.84 | 0.89 | 0.88 | 0.88 | 0.78 | 0.79 | 0.86 | 0.86 | 0.82 | 0.65 |
| site | Total N | 0.78 | 0.78 | 0.79 | 0.74 | 0.78 | 0.80 | 0.81 | 0.72 | 0.75 | 0.84 | 0.84 | 0.82 | 0.62 |
| site | Olsen P | 0.01 | −0.00 | −0.01 | 0.00 | 0.00 | 0.02 | 0.08 | 0.01 | 0.01 | 0.04 | 0.03 | 0.05 | −0.01 |
| site | Exch. Ca | 0.69 | 0.72 | 0.72 | 0.71 | 0.71 | 0.71 | 0.74 | 0.65 | 0.66 | 0.72 | 0.73 | 0.72 | 0.59 |
| site | Exch. Mg | 0.57 | 0.56 | 0.57 | 0.58 | 0.60 | 0.45 | 0.48 | 0.34 | 0.46 | 0.55 | 0.57 | 0.49 | 0.50 |
| site | Exch. K | 0.14 | 0.14 | 0.16 | 0.19 | 0.17 | 0.17 | 0.18 | 0.17 | 0.15 | 0.17 | 0.15 | 0.13 | 0.14 |
| site | eCEC | 0.81 | 0.81 | 0.81 | 0.82 | 0.83 | 0.80 | 0.82 | 0.78 | 0.82 | 0.79 | 0.82 | 0.80 | 0.75 |
| site | Total S | 0.14 | 0.15 | 0.10 | 0.09 | 0.36 | 0.13 | 0.19 | 0.15 | 0.13 | 0.16 | 0.12 | 0.15 | 0.11 |
| site | Total Fe | 0.72 | 0.62 | 0.66 | 0.66 | 0.68 | 0.54 | 0.66 | 0.61 | 0.59 | 0.59 | 0.54 | 0.58 | 0.49 |
| site | Total Mn | 0.45 | 0.44 | 0.39 | 0.38 | 0.35 | 0.33 | 0.10 | 0.41 | 0.43 | 0.39 | 0.44 | 0.36 | 0.15 |
| site | Total Zn | 0.53 | 0.49 | 0.54 | 0.55 | 0.60 | 0.51 | 0.53 | 0.60 | 0.59 | 0.42 | 0.41 | 0.45 | 0.38 |
| site | Total Cu | 0.21 | 0.30 | 0.39 | 0.36 | 0.27 | 0.22 | 0.27 | 0.40 | 0.38 | 0.21 | 0.31 | 0.36 | 0.30 |
| aez | pH | 0.64 | 0.72 | 0.85 | 0.85 | 0.82 | 0.82 | 0.79 | 0.74 | 0.76 | 0.83 | 0.79 | 0.72 | 0.69 |
| aez | Organic C | 0.81 | 0.84 | 0.79 | 0.79 | 0.84 | 0.80 | 0.87 | 0.71 | 0.70 | 0.84 | 0.84 | 0.75 | 0.61 |
| aez | Total N | 0.71 | 0.75 | 0.77 | 0.75 | 0.77 | 0.74 | 0.83 | 0.72 | 0.73 | 0.84 | 0.82 | 0.77 | 0.53 |
| aez | Olsen P | 0.03 | 0.02 | −0.03 | −0.02 | 0.02 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.05 | 0.05 | −0.02 |
| aez | Exch. Ca | 0.66 | 0.64 | 0.62 | 0.61 | 0.72 | 0.67 | 0.71 | 0.69 | 0.65 | 0.58 | 0.69 | 0.62 | 0.51 |
| aez | Exch. Mg | 0.45 | 0.58 | 0.49 | 0.48 | 0.63 | 0.62 | 0.60 | 0.50 | 0.56 | 0.54 | 0.57 | 0.47 | 0.48 |
| aez | Exch. K | 0.08 | 0.11 | 0.15 | 0.14 | 0.13 | 0.02 | 0.11 | 0.13 | 0.16 | 0.07 | 0.13 | 0.14 | 0.12 |
| aez | eCEC | 0.78 | 0.69 | 0.67 | 0.79 | 0.80 | 0.76 | 0.79 | 0.75 | 0.69 | 0.74 | 0.73 | 0.76 | 0.67 |
| aez | Total S | 0.01 | 0.12 | 0.09 | 0.11 | 0.39 | 0.15 | 0.20 | 0.19 | 0.17 | 0.11 | 0.08 | 0.06 | 0.11 |
| aez | Total Fe | 0.49 | 0.53 | 0.61 | 0.60 | 0.63 | 0.49 | 0.59 | 0.47 | 0.51 | 0.60 | 0.55 | 0.47 | 0.40 |
| aez | Total Mn | 0.33 | 0.37 | 0.39 | 0.45 | 0.45 | 0.35 | 0.39 | 0.41 | 0.40 | 0.33 | 0.31 | 0.37 | 0.16 |
| aez | Total Zn | 0.34 | 0.32 | 0.26 | 0.30 | 0.32 | 0.28 | 0.29 | 0.37 | 0.35 | 0.27 | 0.23 | 0.35 | 0.21 |
| aez | Total Cu | 0.02 | 0.17 | 0.22 | 0.06 | 0.08 | −0.03 | 0.17 | 0.31 | 0.35 | 0.16 | 0.16 | 0.05 | 0.23 |
| country | pH | 0.63 | 0.67 | 0.83 | 0.85 | 0.84 | 0.75 | 0.79 | 0.63 | 0.65 | 0.75 | 0.67 | 0.68 | 0.42 |
| country | Organic C | 0.82 | 0.82 | 0.71 | 0.72 | 0.80 | 0.78 | 0.84 | 0.66 | 0.70 | 0.83 | 0.78 | 0.70 | 0.57 |
| country | Total N | 0.69 | 0.70 | 0.72 | 0.68 | 0.72 | 0.67 | 0.76 | 0.61 | 0.64 | 0.82 | 0.75 | 0.68 | 0.45 |
| country | Olsen P | 0.02 | 0.01 | −0.02 | 0.00 | 0.02 | −0.03 | 0.03 | 0.02 | 0.02 | −0.08 | −0.00 | 0.00 | −0.03 |
| country | Exch. Ca | 0.70 | 0.67 | 0.75 | 0.63 | 0.74 | 0.68 | 0.74 | 0.71 | 0.72 | 0.74 | 0.75 | 0.72 | 0.64 |
| country | Exch. Mg | 0.33 | 0.44 | 0.37 | 0.31 | 0.31 | 0.30 | 0.24 | 0.30 | 0.25 | 0.35 | 0.40 | 0.39 | 0.18 |
| country | Exch. K | 0.06 | 0.17 | 0.11 | 0.15 | 0.05 | 0.14 | 0.10 | 0.11 | 0.13 | −0.01 | 0.14 | 0.13 | 0.10 |
| country | eCEC | 0.66 | 0.70 | 0.75 | 0.74 | 0.78 | 0.72 | 0.77 | 0.72 | 0.75 | 0.77 | 0.79 | 0.74 | 0.69 |
| country | Total S | 0.09 | 0.14 | 0.08 | 0.04 | 0.09 | 0.13 | 0.09 | 0.07 | 0.04 | 0.13 | −0.45 | 0.12 | 0.10 |
| country | Total Fe | 0.60 | 0.63 | 0.62 | 0.65 | 0.65 | 0.52 | 0.62 | 0.60 | 0.61 | 0.61 | 0.44 | 0.36 | 0.40 |
| country | Total Mn | 0.34 | 0.22 | 0.27 | 0.28 | 0.26 | 0.08 | 0.10 | 0.39 | 0.33 | 0.17 | 0.14 | 0.07 | 0.07 |
| country | Total Zn | 0.46 | 0.44 | 0.47 | 0.42 | 0.41 | 0.53 | 0.42 | 0.49 | 0.43 | 0.45 | 0.46 | 0.46 | 0.30 |
| country | Total Cu | 0.31 | 0.32 | 0.34 | 0.35 | 0.39 | 0.39 | 0.36 | 0.28 | 0.22 | 0.14 | 0.14 | 0.19 | −0.13 |


Table S4: Agreement between the round-one NumPy implementations and scikit-learn re-implementations run under the identical round-one protocol (round-one preprocessing, same three configurations, fold 0 for selection, folds 1 to k−1 evaluated, spectra only). Fold R² values below −3 were truncated at −3 for the means and correlation.

| Design | Model | Fold × property pairs | Mean fold R² (round-1 NumPy) | Mean fold R² (library) | Median |difference| | Correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cluster | GradientBoosting | 52 | 0.732 | 0.731 | 0.013 | 0.99 |
| cluster | PLSR | 52 | 0.539 | 0.636 | 0.079 | 0.45 |
| cluster | RandomForest | 52 | 0.695 | 0.695 | 0.009 | 0.99 |
| cluster | Ridge | 52 | 0.685 | 0.685 | 0.000 | 1.00 |
| country | GradientBoosting | 39 | 0.205 | 0.213 | 0.037 | 0.99 |
| country | PLSR | 39 | −0.311 | −0.236 | 0.301 | 0.74 |
| country | RandomForest | 39 | 0.171 | 0.187 | 0.041 | 0.98 |
| country | Ridge | 39 | 0.042 | 0.042 | 0.000 | 1.00 |
| site | GradientBoosting | 91 | 0.351 | 0.325 | 0.027 | 0.97 |
| site | PLSR | 91 | −0.037 | 0.193 | 0.177 | 0.73 |
| site | RandomForest | 91 | 0.306 | 0.290 | 0.033 | 0.98 |
| site | Ridge | 91 | 0.180 | 0.180 | 0.000 | 1.00 |


Table S5: Sensitivity of mean leave-site-out pooled R² (13 properties) to spectral preprocessing, with nested tuning as in the main analysis.

| Preprocessing | KRR | PLSR |
| --- | ---: | ---: |
| SG1+SNV, CO2 masked (adopted) | 0.520 | 0.492 |
| SG1+SNV, masked, bin x8 | 0.526 | 0.496 |
| SG1+SNV, unmasked (round 1) | 0.528 | 0.495 |
| SG2+SNV, masked | 0.474 | 0.372 |
| SNV only, masked | 0.558 | 0.424 |
| SNV then SG1, masked | 0.528 | 0.459 |


Table S6: Mean Spearman correlation between importance profiles of the leave-site-out hybrid models (eight folds): DAFM attention spread over receptive fields, occlusion, integrated gradients (IG) and PLSR variable importance in projection (VIP).

| Property | Attention–occlusion | Attention–IG | Attention–VIP | Occlusion–IG | Occlusion–VIP | IG–VIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exch. Ca | 0.39 | 0.88 | 0.27 | 0.38 | 0.13 | 0.30 |
| Total Cu | 0.45 | 0.84 | 0.05 | 0.43 | 0.01 | 0.05 |
| Total Fe | 0.52 | 0.75 | 0.16 | 0.54 | 0.13 | 0.23 |
| Exch. K | 0.12 | 0.79 | 0.23 | 0.13 | 0.01 | 0.28 |
| Exch. Mg | 0.33 | 0.82 | 0.17 | 0.29 | 0.02 | 0.23 |
| Total Mn | 0.34 | 0.76 | 0.16 | 0.36 | 0.10 | 0.20 |
| Organic C | 0.48 | 0.87 | 0.12 | 0.50 | 0.15 | 0.16 |
| Olsen P | 0.32 | 0.76 | 0.10 | 0.26 | 0.05 | 0.11 |
| Total S | 0.35 | 0.88 | 0.19 | 0.36 | 0.06 | 0.21 |
| Total N | 0.56 | 0.89 | 0.11 | 0.54 | 0.07 | 0.12 |
| Total Zn | 0.33 | 0.72 | 0.07 | 0.31 | 0.14 | 0.11 |
| eCEC | 0.46 | 0.86 | 0.22 | 0.43 | 0.08 | 0.29 |
| pH | 0.44 | 0.83 | 0.21 | 0.47 | 0.20 | 0.26 |


Table S7: Coverage of group-calibrated split-conformal 90% intervals by property, design and model.

| Property | aez–Hybrid | aez–KRR | aez–LightGBM | coun–Hybrid | coun–KRR | coun–LightGBM | site–Hybrid | site–KRR | site–LightGBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exch. Ca | 0.77 | 0.78 | 0.80 | 0.88 | 0.79 | 0.89 | 0.79 | 0.78 | 0.79 |
| Total Cu | 0.89 | 0.85 | 0.90 | 0.73 | 0.80 | 0.73 | 0.91 | 0.91 | 0.91 |
| Total Fe | 0.88 | 0.82 | 0.87 | 0.90 | 0.90 | 0.93 | 0.80 | 0.86 | 0.84 |
| Exch. K | 0.86 | 0.82 | 0.86 | 0.87 | 0.84 | 0.85 | 0.85 | 0.88 | 0.87 |
| Exch. Mg | 0.80 | 0.81 | 0.78 | 0.76 | 0.74 | 0.79 | 0.79 | 0.83 | 0.80 |
| Total Mn | 0.90 | 0.92 | 0.89 | 0.82 | 0.83 | 0.82 | 0.73 | 0.77 | 0.76 |
| Organic C | 0.86 | 0.83 | 0.82 | 0.68 | 0.76 | 0.69 | 0.84 | 0.86 | 0.90 |
| Olsen P | 0.93 | 0.92 | 0.93 | 0.81 | 0.87 | 0.87 | 0.89 | 0.89 | 0.90 |
| Total S | 0.89 | 0.86 | 0.88 | 0.75 | 0.79 | 0.78 | 0.87 | 0.87 | 0.84 |
| Total N | 0.85 | 0.87 | 0.91 | 0.73 | 0.77 | 0.80 | 0.85 | 0.84 | 0.89 |
| Total Zn | 0.81 | 0.83 | 0.79 | 0.82 | 0.87 | 0.79 | 0.86 | 0.83 | 0.84 |
| eCEC | 0.81 | 0.81 | 0.79 | 0.81 | 0.79 | 0.84 | 0.86 | 0.84 | 0.84 |
| pH | 0.91 | 0.92 | 0.88 | 0.72 | 0.80 | 0.81 | 0.87 | 0.84 | 0.87 |


Table S8: Learning curves: mean pooled leave-site-out R² (13 properties) for training sets of n samples drawn from m training sites (three draws). "all" denotes all training sites with at least 20 samples (11–13 sites, depending on the fold).

| Model | Training sites | n = 100 | n = 200 | n = 400 | n = 800 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLSR | 2 | −0.285 | −0.537 | – | – |
| PLSR | 3 | −0.349 | −0.014 | – | – |
| PLSR | 4 | 0.217 | 0.261 | 0.238 | – |
| PLSR | 6 | 0.267 | 0.380 | 0.408 | – |
| PLSR | 8 | 0.382 | 0.423 | 0.464 | 0.457 |
| PLSR | 10 | 0.377 | 0.427 | 0.472 | 0.499 |
| PLSR | all | 0.379 | 0.439 | 0.467 | 0.495 |
| KRR | 2 | 0.142 | 0.041 | – | – |
| KRR | 3 | 0.253 | 0.246 | – | – |
| KRR | 4 | 0.304 | 0.346 | 0.280 | – |
| KRR | 6 | 0.345 | 0.408 | 0.449 | – |
| KRR | 8 | 0.407 | 0.457 | 0.458 | 0.478 |
| KRR | 10 | 0.412 | 0.456 | 0.456 | 0.507 |
| KRR | all | 0.422 | 0.467 | 0.505 | 0.532 |
| Cubist | 2 | 0.106 | −0.057 | – | – |
| Cubist | 3 | 0.243 | 0.224 | – | – |
| Cubist | 4 | 0.305 | 0.352 | 0.163 | – |
| Cubist | 6 | 0.343 | 0.417 | 0.416 | – |
| Cubist | 8 | 0.382 | 0.449 | 0.450 | 0.456 |
| Cubist | 10 | 0.422 | 0.437 | 0.441 | 0.473 |
| Cubist | all | 0.416 | 0.449 | 0.469 | 0.496 |

