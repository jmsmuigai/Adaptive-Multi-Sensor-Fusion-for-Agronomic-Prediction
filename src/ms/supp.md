**Supplementary Material**

**Comparison of a Hybrid Deep Learning Model with Machine Learning Methods for Mid-Infrared Soil Nutrient Prediction under Spatial Transfer in East Africa**

Table S1: Composition of the outer folds of each validation design. Site names are followed by country and sample count.

[[TABLE:S1]]

Table S2: Hyperparameter search spaces. Selection always used training-fold data only, with a group-disjoint inner split or inner folds (Section 2.6). Neural configurations list learning rate (lr), convolution channels (c1, c2), first kernel size (k1) and stride (s1), hidden widths (h, h1, h2), RBF units (R), dropout and weight decay (wd); unspecified values are as in BASE: c1 = 16, c2 = 32, k1 = 11, s1 = 2, k2 = 7, attention dimension 32, ancillary embedding 8, RBF projection 8.

[[TABLE:S2]]

Table S3: Pooled out-of-fold R² for every model, property and validation design (seed mean for neural models; predictions limited to the training range). RMSE, RPD, RPIQ, CCC, MAE skill and log-scale R² for every combination are provided in the released results file (pooled_metrics_all.csv).

[[TABLE:S3]]

Table S4: Agreement between the round-one NumPy implementations and scikit-learn re-implementations run under the identical round-one protocol (round-one preprocessing, same three configurations, fold 0 for selection, folds 1 to k−1 evaluated, spectra only). Fold R² values below −3 were truncated at −3 for the means and correlation.

[[TABLE:S4]]

Table S5: Sensitivity of mean leave-site-out pooled R² (13 properties) to spectral preprocessing, with nested tuning as in the main analysis.

[[TABLE:S5]]

Table S6: Mean Spearman correlation between importance profiles of the leave-site-out hybrid models (eight folds): DAFM attention spread over receptive fields, occlusion, integrated gradients (IG) and PLSR variable importance in projection (VIP).

[[TABLE:S6]]

Table S7: Coverage of group-calibrated split-conformal 90% intervals by property, design and model.

[[TABLE:S7]]

Table S8: Learning curves: mean pooled leave-site-out R² (13 properties) for training sets of n samples drawn from m training sites (three draws). "all" denotes all training sites with at least 20 samples (11–13 sites, depending on the fold).

[[TABLE:S8]]
