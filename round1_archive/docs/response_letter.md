# Response to Reviewers

**Manuscript ID:** 1231
**Title (revised):** Adaptive Multi-Modal Fusion for Agronomic Prediction: An Attention-Weighted Hybrid CNN–RBF Model for Soil Nutrient Prediction Across East African Agro-Ecological Zones
**Journal:** Journal of Agriculture, Science and Technology (JAGST)
**Corresponding author:** James Mukoma Mburu (jmsmuigai@gmail.com)

---

To the Editorial Committee,
Journal of Agriculture, Science and Technology

Dear Editorial Committee,

We thank the reviewer for a critique that was exacting, specific and, on every substantive point, correct. The original manuscript contained foundational problems in target construction, data provenance, architecture nomenclature and validation design. We do not dispute any of them.

The reviewer's judgement was that the paper required "a major methodological reconstruction and a fully auditable results pipeline." We agree, and that is what we have done. We did not revise the text around the previous results. We audited the original pipeline, found the defects the reviewer suspected, discarded the dataset, and rebuilt the study from first principles on an openly licensed, independently documented source.

We set out below what the audit found, what we changed, and how each of the sixteen major comments is addressed.

---

## What the audit of the original pipeline found

Three findings determined the scope of the reconstruction, and we report them plainly because they bear on how the reviewer's concerns were resolved.

**The target variable was dimensionally invalid.** The original nutrient gap was computed as a fixed per-nutrient constant minus a measured soil concentration. Inspection of the source data showed the constants were crop requirement rates expressed per unit area, whereas the measurements were concentrations expressed per unit mass. The subtraction was therefore not dimensionally consistent. Because the subtrahend was a single constant per nutrient, the derived variable was also an affine transformation of concentration, yielding identical R², correlations and feature rankings.

**The sign convention was inverted relative to the implemented formula.** Verified against all 85,184 rows of the source file, the implemented formula was Required − Available. Under that formula, and given that measured concentrations greatly exceeded the constants, negative values indicate surplus rather than deficiency. The agronomic interpretation in the original manuscript was therefore reversed, exactly as the reviewer suspected in Major Comment 2.

**The sample coordinates could not be reconciled with the stated study area.** The coordinates supplied with the original dataset placed the samples outside East Africa, in conflict with the accompanying bioclimatic values. We could not establish a defensible provenance for them.

**Several figures were not generated from model output.** The figure-generation script constructed the study-area map, the correlation matrix, the predicted-versus-observed scatter, the training curves and the SHAP panels from simulated arrays rather than from computed results. We withdraw those figures without reservation. Every figure in the revised manuscript is generated from the computed pipeline outputs, and the script that produces them is released.

We are grateful the reviewer pressed on these points. Publishing the original manuscript would have been a serious error.

---

## The reconstructed study

The revised manuscript uses two openly licensed Africa Soil Information Service (AfSIS) Phase I deposits, joined on the soil sample number:

- **Mid-infrared spectra** — Vågen et al. (2020), doi:10.34725/DVN/QXCWP1, measured on a Bruker Tensor 27/HTs-XT FT-IR at the ICRAF Soil-Plant Spectral Diagnostics Laboratory, Nairobi.
- **Reference wet chemistry** — Vågen et al. (2021), doi:10.34725/DVN/66BFOB, analysed at Rothamsted Research.

The analysis set comprises **1,310 georeferenced samples from 16 LDSF sentinel sites across Kenya, Ethiopia, Tanzania and Uganda**, with 207 sampling clusters, 1,749 wavenumbers per sample and thirteen laboratory-measured soil properties. **The join is complete with zero attrition**, which we state explicitly in Section 2.2.

The proposed architecture is retained, because the reviewer's objections concerned its description and evaluation rather than its design. It has been renamed, its mathematics completed, and its evaluation rebuilt.

---

## Point-by-point response

### Major Comment 1 — Title, abstract and contribution claims

**Reviewer:** The terms "SVM-CNN", "dual-branch", "multi-sensor fusion", "high-resolution", "operationally viable" and "directly deployable" are stronger than the evidence presented.

**Response.** Every one of these terms has been removed or substantiated.

"SVM-CNN" is gone; the model is a Hybrid CNN–RBF (see Major Comment 7). "Multi-sensor" has been changed to "multi-modal" throughout the title and text, because the inputs are MIR spectra fused with sampling depth and position rather than multiple remote sensors. "Dual-branch" is now accurate and demonstrated: the architecture has two distinct branches with a learned gate between them, and Table 11 reports the measured contribution of each. "High-resolution", "operationally viable" and "directly deployable" have been deleted. Computational cost is reported as measured latency on stated hardware (Table 10), with an explicit statement that these figures cover inference only and support no claim of deployment readiness.

The abstract now reports the validation design, the confidence intervals, and the finding that the proposed architecture ranks fifth of six within landscapes.

### Major Comment 2 — Response variable: nutrient-gap sign convention

**Reviewer:** This is the most serious inconsistency in the manuscript. A notation correction alone is insufficient unless the numerical pipeline is independently verified.

**Response.** We agree that a notation change would have been inadequate, and we did not make one. We verified the implemented formula against the full source file and confirmed the sign inversion the reviewer identified.

The gap variable has been removed from the study. The models now predict **the raw measured soil properties directly** — pH, organic carbon, total nitrogen, Olsen phosphorus, exchangeable Ca, Mg, K, effective CEC, and total S, Fe, Mn, Zn, Cu — in their original measurement units (Table 2). Agronomic status is evaluated separately in Section 3.7 by comparing measured concentrations against published critical levels, with the deficient fraction reported at the reference threshold and at ±50% of it, so no conclusion depends on a single contested value.

### Major Comment 3 — Construction and agronomic validity of targets

**Reviewer:** Provide the crop, target yield, rooting depth, analytical extraction method, unit conversion, critical level and source for every nutrient.

**Response.** The question no longer arises for the modelling targets, because no requirement value enters the model. Table 2 gives, for every property, the source column, the measurement unit, the number of non-missing observations and the distributional skew. Section 2.2 states the extraction method and instrument for each analyte group: dry combustion for C and N, Olsen extraction for P, cobalt hexamine with ICP-OES for exchangeable cations, and aqua regia digest with ICP-OES for quasi-total elements. Sampling depths are stated (0–20 cm and 20–50 cm) and depth is carried as a model input.

Critical levels used in the post-hoc agronomic assessment are identified as indicative reference values for tropical arable topsoil rather than locally calibrated thresholds, and the sensitivity of every prevalence figure to that choice is reported.

### Major Comment 4 — Data provenance, dates and sample accounting

**Reviewer:** Provide a complete data audit. Report the exact dataset version and persistent identifier, collection dates, coordinates, depth, laboratory methods, inclusion flow, duplicates, missingness and per-zone counts.

**Response.** Section 2.2 and Table 1 provide this in full. Both deposits are cited with DOIs and version numbers. Collection dates (2009–2013) and sampling protocol are documented and cited. Per-country sample counts, sentinel-site counts, cluster counts and coordinate ranges are tabulated. Depth counts are given (659 topsoil, 651 subsoil). Missing values are reported per property. Duplicate identifiers were checked and none were found. The join between the two deposits is reported with its attrition, which is zero.

An audit file recording all of these quantities is released with the pipeline and is regenerated automatically whenever the dataset is rebuilt.

### Major Comment 5 — Predictor provenance, leakage and operational realism

**Reviewer:** Clarify whether soil properties are measured at the sampled points or extracted from SoilGrids, and rule out circularity.

**Response.** The concern is eliminated by the change of predictor. The model's primary input is the **mid-infrared spectrum of the sample itself**, a direct physical measurement on the soil being predicted. No mapped soil product enters the model, so there is no possibility of lineage overlap between predictors and targets. The only ancillary inputs are sampling depth and geographic position, both recorded in the field.

### Major Comment 6 — Spatial and temporal harmonisation

**Reviewer:** Upsampling coarse products to a 10 m grid does not create 10 m information. The claim that one 1 km pixel becomes 100 ten-metre sub-pixels is mathematically incorrect.

**Response.** The reviewer is correct on both counts, and the arithmetic error is acknowledged: a 1 km pixel subdivides into 100 × 100 = 10,000 cells at 10 m, not 100. All resolution claims have been withdrawn along with the satellite covariate stack. Spectral preprocessing is now described completely and reproducibly in Section 2.3: Savitzky–Golay first derivative with stated window and polynomial order, Standard Normal Variate scatter correction, binning factor, and per-band standardisation, with the explicit statement that every data-dependent step is fitted within the training fold only.

An Earth Engine script for covariate extraction is nevertheless released with the package for future work. It documents the exact collection identifier, bands, scale factors, quality bitmasks and compositing window of every product, together with a table of native spatial support, and it uses Landsat 7 rather than Sentinel-2 because the latter did not launch until 2015 and cannot be contemporaneous with 2009–2013 sampling. That correction was prompted by the reviewer's insistence on exact product identifiers.

### Major Comment 7 — Architecture nomenclature and mathematical validity

**Reviewer:** A dense network with a custom RBF activation is not an SVR. Either implement an SVR-informed objective or rename the component.

**Response.** We accept the criticism entirely and have renamed the component. All references to SVM and SVR are removed. Section 2.4.4 now states explicitly that the head "is **not** a support vector regressor. It has no ε-insensitive loss, no margin, no dual coefficients and no support vectors. It is a radial-basis-activated neural regression head, and is named as such throughout."

We have also completed the mathematics the reviewer found lacking. Section 2.4.5 adds three propositions with proofs: that the attention mechanism performs convex pooling and therefore cannot produce a context vector outside the range of the features it aggregates; that the RBF head's output is bounded by ‖**W**_o‖₁ regardless of input; and that its gradient saturates for inputs far from the calibration domain. Table 6 shows the empirical consequence — models with bounded output occupy the three least severe worst-case failures and models with unbounded output the three most severe, exactly as the bound predicts.

### Major Comment 8 — DAFM formulation and interpretability

**Reviewer:** Latent dimensions cannot be relabelled as input variables. Define every tensor shape and explain how input-level importance was derived.

**Response.** The reviewer identified a genuine invalidity, and we redesigned the module rather than re-describing it. Attention now operates over **spectral positions rather than latent channels** (Section 2.4.2), so each weight corresponds to a contiguous wavenumber window and is attributable to an identifiable absorbance region. Every tensor shape is stated in the class documentation and in Equations 1–9. Attention weights are plotted directly against the mean spectrum with functional-group regions annotated (Figure 10), so the reader can judge the correspondence independently.

The previously reported SHAP-concordance figures are withdrawn. In their place, the learned modality gate reports a **measured** quantity: the proportion of each prediction attributable to the spectral branch, with fold-level standard deviations (Table 11). We state explicitly that attention weights describe where the network routes information and are not causal statements.

### Major Comment 9 — Feature ordering, preprocessing and target scaling

**Reviewer:** Justify applying a 1D convolution to a heterogeneous tabular vector. All clipping, imputation, normalisation and tuning must be fitted only within the training fold.

**Response.** This comment prompted the most consequential design change. The reviewer was right that a one-dimensional convolution has no justification over an arbitrarily ordered tabular vector, because local kernels presuppose meaningful adjacency. In the revised study the convolution operates over the **MIR wavenumber axis**, which is a genuinely ordered sequence in which neighbouring bands are physically related through the width of absorbance features. The inductive bias is now appropriate to the data, and Section 2.4.1 makes that argument explicitly.

On preprocessing, Section 2.3 states that every data-dependent step is fitted on training-fold rows alone, and Section 2.9 lists preprocessing leakage among the threats to validity with the control applied. Target transformation is stated explicitly: properties with skew above 2.0 are log-transformed for fitting, and all predictions are back-transformed before any metric is computed, so every reported error is in the original measurement unit.

### Major Comment 10 — Validation design, uncertainty and baseline fairness

**Reviewer:** A random split is vulnerable to spatial autocorrelation. Spatial validation should be the primary analysis, with fair tuning, ablations and uncertainty.

**Response.** Random splitting has been abandoned entirely; it appears nowhere in the revised manuscript. The study now uses **three nested, group-disjoint validation designs** (Section 2.7): leave-cluster-out over 207 sampling clusters, leave-site-out over 16 sentinel sites, and leave-country-out over four national datasets. In every design, fold 0 is reserved for hyperparameter selection and never used as a test fold, so no reported metric is contaminated by model selection.

On fairness, **every model receives exactly three candidate configurations**, scored identically. We agree with the reviewer's implicit point that unequal tuning is a common source of spurious architectural advantage, and Section 2.9 lists it as a threat to validity with the control applied.

On uncertainty, R², RMSE, MAE, bias and normalised RMSE are each reported as fold means with standard deviations and t-based 95% confidence intervals. Mean ranks and counts of properties retaining positive R² are reported alongside means, because a single catastrophic fold can dominate an average — as it does for the multilayer perceptron under leave-site-out.

We also report, as a direct consequence of adopting these designs, that predictive skill declines from mean R² of 0.749 under leave-cluster-out to 0.351 under leave-site-out and 0.205 under leave-country-out. We regard this as the single most useful number in the paper for a reader trying to interpret published accuracies.

### Major Comment 11 — Analyses introduced without methods or results

**Reviewer:** Several findings appear only in the Discussion with no corresponding Methods or Results.

**Response.** All of the specific items cited — the 25th-percentile threshold validation, the ±15/30/45-day compositing sensitivity, the SHAP interaction analysis, the AEZ-holdout transfer pilot, the 90.7% transfer efficiency and the 533-sample recalibration — have been removed. None survived the audit.

Their legitimate successors are fully specified in Methods and reported in Results with uncertainty: cross-zone transfer is now the leave-site-out and leave-country-out designs (Sections 2.7, 3.2, 3.4); threshold sensitivity is reported as prevalence at ±50% of each critical level (Table 12); and interpretability is the attention and gate analysis (Section 3.6). Nothing appears in the Discussion that is not defined in Methods and computed in Results.

### Major Comment 12 — Figures, traceability and visual evidence

**Reviewer:** Figures require a full integrity and readability audit. Identify each curve as training, validation or test. Release raw logs and seeds.

**Response.** As stated above, the original figures were not generated from model output and are withdrawn. All eleven figures in the revised manuscript are produced by a released script that reads only the computed result files and the source data; it contains no simulated arrays. Figures and tables are placed inline at the point of first reference, as the journal requires, rather than collected at the end. Training and validation curves are labelled as such and derive from actual training logs, which are released. The random seed is stated in the manuscript and in the code.

### Major Comment 13 — Results consistency and interpretation

**Reviewer:** Correct internal inconsistencies and temper the inference.

**Response.** Beyond correcting the specific inconsistencies, we have added a structural safeguard. An automated verification script re-derives **every numeric claim in the manuscript** from the stored per-fold records and exits with an error if any value has drifted. It currently performs 109 checks covering dataset composition, per-property and per-model results under all three designs, the calibration-breadth contrast, per-country results, gate weights, deficiency prevalence, computational cost, equation and table numbering, abstract length and reference completeness. All 109 pass. The script is released so that any reader can run the same audit.

Inference has been tempered throughout. The paper now reports that the proposed architecture ranks fifth of six within landscapes and mid-pack under transfer, and states plainly in Section 3.1 that "we had anticipated that convolutional treatment of the ordered spectral sequence... would confer an advantage in this regime, and it did not."

### Major Comment 14 — Computational efficiency and deployment claims

**Reviewer:** Report deployment metrics reproducibly: exact hardware, software versions, repetitions, percentile latency. Benchmark all models on the same hardware.

**Response.** Table 10 reports training time, mean and 95th-percentile per-sample inference latency and throughput for every model, measured on the same hardware over 30 timed repetitions following a warm-up pass. The platform, RAM, Python version, NumPy version, sample count, band count and thread configuration are all stated. Section 3.5 notes explicitly that these figures cover model inference only and exclude sample preparation and spectrometer acquisition, and makes no claim of operational readiness.

### Major Comment 15 — Reproducibility and data availability

**Reviewer:** "Available from the corresponding author upon reasonable request" does not permit independent verification.

**Response.** We have withdrawn that statement. Section 7.0 now points to a public repository containing the complete pipeline:

**https://github.com/jmsmuigai/Adaptive-Multi-Sensor-Fusion-for-Agronomic-Prediction**

It contains the model implementations, validation designs, preprocessing and metrics, dataset assembly, experiment driver, supporting analyses, figure and table generation, the verification script, the Earth Engine covariate script, and an interactive Google Colab notebook that downloads the data and reproduces the pipeline end to end. Also released are the random seed, fold assignments for all three designs, selected hyperparameter configurations, raw training logs, and per-fold predictions and metrics for all 1,794 model–property–design–fold combinations.

The pipeline is implemented in NumPy without machine-learning framework dependencies, so it reproduces from a bare Python installation and is not subject to version drift in a deep-learning library. Both source datasets are openly licensed and require no account to download.

### Major Comment 16 — References and technical accuracy

**Reviewer:** Audit every citation. The novelty claims require systematic comparison with recent work. Ensure every agronomic threshold is supported by an accessible source.

**Response.** The reference list has been rebuilt. It now contains fifteen entries, each verified against the publisher record and each followed by a resolvable link, as the journal requires. The unverifiable AfSIS technical report has been replaced by the two dataset DOIs actually used.

The literature positioning has been brought current. Section 1.0 situates the work against 2025–2026 attention-based spectroscopy — Guo et al. (2025) on Swin Transformers for MIR, Shuai et al. (2025) on channel attention with ConvNeXt, Peng (2026) on self-attention ensembles — and Section 3.8 compares our results directly with Hengl et al. (2017) and Ng et al. (2019), noting where they agree and where they do not. Unqualified novelty claims have been removed.

---

## Summary of changes

| Change | Section |
|---|---|
| Dataset replaced with two openly licensed AfSIS deposits, cited by DOI | 2.2 |
| Target changed from a derived gap to thirteen directly measured properties | 2.2, Table 2 |
| Architecture renamed; head stated not to be an SVR | 2.4.4 |
| Attention moved from latent channels to spectral positions | 2.4.2 |
| Three propositions with proofs added, bounding the model's failure | 2.4.5 |
| Random splitting abandoned; three nested group-disjoint designs adopted | 2.7 |
| Equal tuning budget for all six models | 2.6 |
| Threats-to-validity section added with the control applied to each | 2.9 |
| All figures regenerated from computed outputs; fabricated figures withdrawn | 3.1–3.7 |
| Figures and tables placed inline at first reference | throughout |
| Measured computational cost on stated hardware | 3.5, Table 10 |
| Public repository and Colab notebook released | 7.0 |
| Automated verification of 109 numeric claims | 7.0 |
| Reference list rebuilt, all entries verified and linked | 8.0 |

---

We recognise that this is a more thorough reconstruction than a revision normally entails, and that the paper's conclusions are correspondingly more modest than those originally submitted. We think it is a better paper for it. Every number is now traceable to a computation a reader can rerun, and the results that do not favour our own architecture are reported as plainly as those that do.

We are grateful to the reviewer for insisting on a standard the original manuscript did not meet.

Sincerely,

**James Mukoma Mburu** (Corresponding Author)
**Kennedy Ogada**
**Michael Kimwele**

Department of Computing, School of Computing and Information Technology
Jomo Kenyatta University of Agriculture and Technology
P.O. Box 62000-00200, Nairobi, Kenya
jmsmuigai@gmail.com
