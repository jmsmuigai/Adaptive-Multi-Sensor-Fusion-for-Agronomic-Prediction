**Cover Letter and Point-by-Point Response to the Reviewer**

**Manuscript ID:** 1283

**Original title:** "Adaptive Multi-Modal Fusion for Agronomic Prediction: An Attention-Weighted Hybrid CNN–RBF Model for Soil Nutrient Prediction Across East African Agro-Ecological Zones"

**Revised title:** "Comparison of a Hybrid Deep Learning Model with Machine Learning Methods for Mid-Infrared Soil Nutrient Prediction under Spatial Transfer in East Africa"

**Running title:** Spectral soil nutrient prediction under spatial transfer

**Journal:** Journal of Agriculture, Science and Technology (JAGST) — Decision: Revisions Required

Dear Editor,

Thank you for the opportunity to revise our manuscript, and thank you to the reviewer for an unusually careful and constructive report. The reviewer identified problems with the experiments, the statistics and several claims. We agreed with nearly all of them, and we have therefore rebuilt the analysis rather than amending the text.

Our response to the editor's instructions is as follows:

> \- **(I) References.** Every reference was checked against its source (Crossref/publisher records). Two round-one references had wrong bibliographic details and have been corrected: Gamagedara *et al.* (2025), whose author list, issue and volume were wrong, and the 2026 *Sensors* paper, which is by Yin, Deng, Li, Feng and Peng, not "Peng (2026)". Thirty-five references requested by the reviewer, or needed for the new analyses, were added. All references are now cited in the text, and all citations are listed.
>
> \- **(II) Track changes.** Two versions are uploaded: one with all changes tracked against the round-one manuscript, and one clean version. Because the revision is extensive, most of the text appears as tracked changes.
>
> \- **(III–IV) Point-by-point responses.** These follow below. We were able to address every comment; where our new results did not support a round-one claim, we have withdrawn the claim and say so explicitly.
>
> \- **(VI) English.** The whole manuscript has been rewritten and edited for clarity and consistent technical usage.
>
> \- **(IX) Anonymity.** Author names, affiliations, institutional acknowledgements and the repository link (which contained a user name) have been removed from the manuscript and from the file metadata. The repository will be made public on acceptance.

We must also report three faults that we found in our own round-one code while re-implementing the analysis. First, the multilayer perceptron and the hybrid early-stopped on the **test fold**; the round-one text wrongly said that a held-out part of the training fold was used. Second, the NumPy PLSR computed its regression coefficients incorrectly (without the weight matrix), which depressed its scores. Third, hyperparameters were selected on a single reserved fold. None of the round-one numbers is carried forward. Every number in the revised manuscript is regenerated from stored per-fit records by one script, and the round-one baselines were re-run with library implementations to quantify the effect (new Table S4).

The main consequences for the paper are:

> 1\. The proposed hybrid does **not** outperform the benchmarks. It ranked last in every design. The ablation shows that its RBF head and modality gate add no skill. We now say this plainly and have changed the title and framing accordingly.
>
> 2\. The paper's contribution is the evaluation itself: nested group-disjoint validation across five spatial scales, ablation, multi-seed training, group-bootstrap inference, a like-for-like test of calibration breadth, local spiking, group-aware conformal intervals and validated interpretability. These give clear, practically useful conclusions for soil spectral libraries in East Africa.

**Summary of the seven experiments requested by the reviewer**

| No. | Reviewer request | What we did | Main result | Where |
|---|---|---|---|---|
| 1 | Local spiking, k = 0, 5, 10, 20, 50 | Added k samples from the held-out site to training, under leave-site-out; profile partners of spiked samples removed from the test set | Mean R² of kernel ridge regression rose from 0.512 (k = 0) to 0.658 (k = 50); Olsen P stayed near 0.06 | Section 3.6, Table 8, Fig. 8 |
| 2 | Learning curve over samples and sites | Crossed training size n = 100–800 with the number of training sites m = 2 to all; plus a like-for-like Kenya test | At n = 200, R² rose from 0.04 (2 sites) to 0.47 (all sites); quadrupling n added only 0.06 | Section 3.5, Table 7, Fig. 7, Table S8 |
| 3 | Ablation table | Nine single-component replacements of the hybrid under leave-site-out, all 13 properties | RBF head and gate gave no benefit; positional information and attention pooling were essential | Section 3.3, Table 5, Fig. 6 |
| 4 | Direct boundedness test, ‖W~o~‖₁, log and original scales | Hybrid compared with an otherwise identical linear-head network; ‖w~o~‖₁ reported | Median ‖w~o~‖₁ 5.49 (loose bound); worst-case R² similar on both scales; boundedness claim withdrawn | Section 3.4, Table 6 |
| 5 | Group-bootstrap CIs and paired tests | 2,000-draw group bootstrap over spatial units; paired tests for all model pairs; Friedman and Nemenyi tests | GPR ranked first; eleven benchmarks within the critical difference; the hybrid outside it | Sections 2.8 and 3.2, Table 4, Table S3 |
| 6 | 5–10 seeds for all neural models | Five seeds (11, 22, 33, 44, 55) for the MLP, CNN, multi-output CNN and hybrid in every fold | Hybrid had the largest seed variability (SD of R² 0.041) | Section 2.6 |
| 7 | Leave-AEZ-out and group-aware conformal prediction | Five Köppen–Geiger zones withheld in turn; split conformal intervals with a group-aware calibration set | Coverage of nominal 90% intervals 84.5% (site), 85.4% (AEZ), 80.5% (country), against 71.7–78.5% with random calibration | Sections 2.1 and 3.7, Tables 3 and 9, Fig. 9 |

**Changes to the title and format**

The title has been changed to "Comparison of a Hybrid Deep Learning Model with Machine Learning Methods for Mid-Infrared Soil Nutrient Prediction under Spatial Transfer in East Africa". The original title promised an adaptive multi-modal model that improves prediction; the re-analysis does not support either claim, and the reviewer noted that "multi-modal fusion" overstated the inputs. The new title follows the descriptive style of comparison studies published in JAGST, states the main comparison and the validation setting, and avoids undefined abbreviations. The running title (seven words) is "Spectral soil nutrient prediction under spatial transfer".

The manuscript now follows the JAGST Author Guidelines and submission checklist: Calibri 11 pt, single line spacing, an abstract of fewer than 300 words with every abbreviation defined, a running title at the foot of the title page, Harvard in-text citations with multiple citations in alphabetical order separated by semicolons, an alphabetical reference list starting on a new page with a link (DOI or URL) for every reference, tables and figures placed in the text where they are first discussed, metric units, and an Ethical Statement (Section 6.0). The text has been checked for originality: it was written afresh for this revision, and quotations from other works are not used.

We hope the revised manuscript now meets the standard of JAGST. We would be glad to make any further changes.

Yours sincerely,

The Authors

**Point-by-point response**

Reviewer comments are in *italics*; our responses follow. Section, table and figure numbers refer to the revised manuscript.

***General comment.*** *The great contribution of the paper is the nested group-disjoint validation … the claims (calibration breadth beats architecture, boundedness explains robustness, attention and gate are interpretable, widespread P deficiency) each have a confound or a validity problem.*

**Response:** We agree. Each claim has been tested directly. The results are:

> \- Breadth versus architecture: restated with like-for-like evidence (Section 3.5).
>
> \- Boundedness: withdrawn (Section 3.4).
>
> \- Interpretability: attention is now validated against occlusion, integrated gradients and PLSR VIP, and the gate is no longer interpreted (Section 3.8).
>
> \- P deficiency prevalence: withdrawn and replaced by screening accuracy (Section 3.9).

The nested validation is now the centre of the paper, extended to five designs: random, cluster, site, agro-ecological zone and country.

**1. Architecture claims not supported (reviewer section 2.1)**

*1.1 No ablations: (a) linear or MLP head, (b) attention versus mean or max pooling, (c) gate versus concatenation, (d) spectra-only versus spectra plus ancillary, (e) RBF head versus dense head.*

**Response:** Done (Section 2.9, Section 3.3, Table 5, Figure 6). Nine single-component replacements were run under leave-site-out with three seeds each (five for the linear head), all 8 folds and 13 properties:

> \- (a/e) Linear head: ΔR² = +0.015. Dense ReLU head: +0.016. Round-one Gaussian head: −0.010. The RBF head therefore gives no benefit.
>
> \- (b) Mean pooling: −0.217, harming all 13 properties. We used mean rather than max pooling as the comparator because, like attention, it pools over positions. We also tested attention re-weighting followed by flattening (+0.008) and removal of the positional embedding (−0.192).
>
> \- (c) Concatenation instead of the gate: +0.013. The gate therefore gives no benefit.
>
> \- (d) Spectra only: −0.039, with significant losses for 5 properties.
>
> \- Adding coordinates: −0.011.

The statement that "the architecture achieves its principal design objective" has been removed. The Conclusion now states that it does not.

*1.2 The "RBF head" is not a kernel method … Intro gap 1 is a straw man … cite deep kernel learning, differentiable GP/SVR layers and RBF layers.*

**Response:** Agreed. Gap 1 has been rewritten (Introduction, first question). The ability to train an RBF or kernel component jointly with an encoder is now presented as well established, citing Park and Sandberg (1991), Wilson *et al.* (2016) and van Amersfoort *et al.* (2020). The question we now pose is whether such a head *improves* prediction, which only ablation can answer. The head itself was replaced by a genuine RBF layer with **learned centres**, φᵢ = exp(−γᵢ‖u − cᵢ‖²) (Eq. 3, Section 2.4.4). The round-one Gaussian-activation head is retained only as an ablation. The text states explicitly that the head is not a kernel machine or a support vector regressor.

*1.3 The propositions are trivial … they do not justify "the architecture cannot fail without bound".*

**Response:** Agreed. Propositions 1–3 and the claim that the model "cannot fail without bound" have been removed. The only remaining statement is the one-line bound |ŷ − b~o~| ≤ ‖w~o~‖₁, presented as a property of the construction, not as evidence of robustness, and tested empirically (Section 3.4).

*1.4 Single seed for neural nets.*

**Response:** Every neural model (MLP, CNN, multi-output CNN, hybrid) was trained from **five random initialisations** in every fold of every design. Results are seed means, with the seed standard deviation reported, and seed variability is propagated into all bootstrap intervals (Section 2.6). The hybrid had the largest seed variability (mean SD of R² 0.041, against 0.028 for the CNN and 0.022 for the MLP).

*1.5 Weak tuning; equal budget is not equal fairness.*

**Response:** Tuning is now nested inside each outer training fold, using group-disjoint inner splits (Section 2.6). Search spaces are sized to each model's hyperparameters (Table S2): 7 values for PLSR, 11 for ridge, 9-point two-dimensional grids for SVR and KRR, marginal-likelihood optimisation for GPR, early-stopped tree numbers for LightGBM, and three architectures for each neural model, trained with five seeds. The neural search space is still modest because of the seed and fold multiplicity, and we say so (Limitation vi). However, the plain CNN with this budget performed on a par with the best conventional models, so the hybrid's poor ranking is not plausibly due to under-tuning alone; the ablations identify specific components.

*1.6 Early-stopping leakage.*

**Response:** The reviewer was right, and the problem was worse than suspected: in round one the neural models early-stopped on the **test** fold (a code fault, now disclosed in Section 2.5 and in this letter). Early stopping and configuration choice now use a **group-disjoint inner split**: whole clusters for the cluster design, and whole sites otherwise. The outer test fold is never used for any decision.

*1.7 Self-implemented baselines. Use scikit-learn, LightGBM/XGBoost and PyTorch, and report agreement.*

**Response:** All models were re-implemented with libraries: scikit-learn (PLSR, ridge, SVR, KRR, GPR, random forest), LightGBM, the cubist package, and PyTorch for all neural models. Memory-based learning, which has no maintained Python package, is implemented following Ramirez-Lopez *et al.* (2013). Agreement with round one was measured under exactly the round-one protocol (Table S4):

> \- Ridge regression reproduced exactly.
>
> \- Random forest and gradient boosting agreed closely (median absolute difference in fold R² of 0.023 and 0.022).
>
> \- PLSR did not agree (0.147), which revealed the round-one coefficient error.

The round-one MLP failure (R² = −24.1 for total Fe) is not reproduced as such, but the same failure mode is reproduced and explained in Section 3.4 (log back-transformation).

**2. Calibration breadth (reviewer section 2.2)**

*Kenya-only versus four-country changes sample size, number of sites and diversity at once; test populations differ … add a learning curve varying n and number of sites, and the like-for-like comparison … the "order of magnitude" statement is not supported.*

**Response:** Both experiments were added (Section 3.5, Table 7, Figure 7, Table S8).

> \- **Like-for-like comparison.** The test sets are always the three Kenyan sites. Mean R² was −0.136 when training on the other two Kenyan sites, 0.101 with a size-matched set drawn from about 13 regional sites, 0.108 with non-Kenyan data only, and 0.230 with Kenyan plus regional data. Diversity at constant size improved 9 of 13 properties.
>
> \- **Learning curves.** We crossed n = 100–800 with m = 2 to all training sites. For kernel ridge regression at n = 200, mean R² rose from 0.04 with 2 sites to 0.47 with all sites, whereas quadrupling n with all sites added only 0.06.

The "order of magnitude" sentence has been deleted. As the reviewer suggested, the comparison is now made honestly against the full spread among the twelve benchmarks (0.057 under leave-site-out), and the conclusion is stated cautiously. The round-one Table 6/7 numbering error no longer arises, because all tables have been renumbered.

**3. Boundedness (reviewer section 2.3)**

*The boundedness argument is contradicted by your own table … a likely confound is the log1p back-transform … report ‖W_o‖₁ and test the claim directly … saturating to zero is not abstention.*

**Response:** We agree and have withdrawn the argument (Section 3.4, Table 6).

> \- The learned ‖w~o~‖₁ had a median of 5.49 (5th–95th percentile 3.23–7.96, maximum 10.7) in standardised units, so the bound is loose.
>
> \- Against an otherwise identical linear-head network, the worst-case R² was similar on both the original and the log scale, under leave-site-out, leave-AEZ-out and leave-country-out.
>
> \- The reviewer's explanation is confirmed: without a range safeguard, five model–property combinations had R² below −1, all for total S, which is log-modelled and has skewness 13.9. They involved ridge, kernel and GP models and the MLP, and none involved the hybrid.
>
> \- Limiting back-transformed predictions to the training range removed these failures. This safeguard is now applied identically to all models and is reported transparently.

The "abstention" sentence has been replaced by the reviewer's point that a near-zero Gaussian unit returns a bias-driven prediction without an uncertainty signal. Group-aware conformal intervals now supply that signal (Section 3.7).

**4. Interpretability (reviewer section 2.4)**

*Attention is not explanation … validate with occlusion/perturbation, integrated gradients, PLSR VIP and a permutation test.*

**Response:** Done (Section 3.8, Figure 10, Table S6). Agreement of attention with the other measures (Spearman ρ):

> \- integrated gradients: 0.82
>
> \- occlusion: 0.39
>
> \- PLSR VIP: 0.16

A permutation test with circular shifts, which preserves autocorrelation, showed attention enrichment inside literature absorbance windows of 1.6–1.8 times the null for organic C, total N and pH (combined p = 0.02–0.04). Few single-fold tests were significant, however. Attention is therefore described as a coarse, hypothesis-generating summary, citing Jain and Wallace (2019) and Wiegreffe and Pinter (2019). The phrase "principal contribution" has been removed.

*Resolution: quoting single wavenumbers overclaims precision.*

**Response:** Agreed. The receptive field of each attended position is now reported as about 286–417 cm⁻¹, with positions about 31 cm⁻¹ apart (Section 2.4.1). All single-wavenumber statements have been removed, and attention is described only in terms of broad regions.

*Band assignments; 2454 cm⁻¹ is near the CO₂ region; mask atmospheric regions.*

**Response:** The reviewer is right; the 2454 cm⁻¹ assignment was wrong and has been removed. The CO₂ region (2450–2250 cm⁻¹) is now removed before modelling (Section 2.3). The remaining assignments cite Nguyen *et al.* (1991), Janik *et al.* (1998) and Parikh *et al.* (2014). A preprocessing sensitivity analysis (Table S5) shows that masking changes transfer skill by less than 0.01 R².

*The gate is not an identifiable contribution.*

**Response:** Agreed. All "% spectral contribution" statements and the gate table have been removed. Contributions are now estimated by permutation and by the spectra-only ablation (Section 3.8). For Olsen P and exchangeable K, depth was nearly as informative as the spectra; for organic C and pH it contributed little.

*Ancillary branch uses coordinates … do benchmarks receive the same features? "Multi-modal fusion" overstates.*

**Response:** All models now receive **identical inputs**: the spectrum plus depth (Section 2.5). Coordinates are excluded from the main analysis because of the extrapolation hazard (Meyer and Pebesma, 2021) and are tested as an ablation: they did not help either the hybrid (ΔR² −0.011) or LightGBM (below 0.001). "Multi-modal fusion" has been removed from the title and replaced by "spectral–ancillary fusion" in the text.

**5. Agronomic section (reviewer section 2.5)**

*Total Cu/Zn thresholds are not meaningful; headline percentages are invalid; non-probability sample; RQ4 needs yield data.*

**Response:** Agreed on all points.

> \- All deficiency prevalences for total Cu, Zn, quasi-total S and total N have been withdrawn, with an explanation citing DTPA extraction (Lindsay and Norvell, 1978).
>
> \- No population prevalence is claimed for any property.
>
> \- RQ4 was rewritten. It now asks which properties with established interpretive classes can be **screened** in an unsampled landscape.
>
> \- Table 10 reports sensitivity, specificity and balanced accuracy for pH, organic C, Olsen P, exchangeable Ca, Mg and K, and eCEC against indicative critical levels (Landon, 1991). Screening was reliable for organic C (balanced accuracy 0.91), pH (0.84) and eCEC (0.83), and close to chance for Olsen P (0.58).

**6. Statistics and metrics (reviewer section 2.6)**

*Per-fold R² unstable; add RMSE/SD, RPD, RPIQ, Lin's CCC and MAE relative to a mean baseline.*

**Response:** Metrics are now computed on pooled out-of-fold predictions. We report R², RMSE, RMSE/SD~train~, RPD, RPIQ, CCC, MAE skill against the training-mean baseline, and log-scale R² (Section 2.8, Table 4, Table S3). Per-fold R² is reported only as a range.

*CIs not valid (t-intervals over few folds; interval above 1).*

**Response:** t-intervals have been replaced by a **group bootstrap** (2,000 draws) over the spatial units, averaged over seeds.

*"Best R²" selected post hoc.*

**Response:** The model-level comparison now uses GPR, the model with the best mean rank, throughout Table 4. The per-property maximum is shown only in a separate column marked as descriptive and optimistic.

*No significance tests.*

**Response:** Paired group-bootstrap tests for all model pairs, plus the Friedman test and Nemenyi critical difference, have been added (Section 3.2). All eleven other benchmarks were within the critical difference of GPR; only the hybrid fell outside it. The round-one emphasis on the single-fold Kenya result has been removed from the abstract and text.

*Folds actually evaluated; Ethiopia never tested; state site grouping.*

**Response:** With nested tuning, **every fold is a test fold** in every design. The country design is therefore a genuine four-country test that includes Ethiopia. Table S1 lists the sites in every leave-site-out fold and the composition of all other folds.

*Tuning leakage (single fold); use nested group CV.*

**Response:** Done (see 1.5).

**7. Data and design (reviewer section 3)**

*Time gap between spectra (2009–2013) and chemistry (2017–2018).*

**Response:** This is now discussed in Section 2.2 and Section 3.9, citing Bartlett and James (1980), as a plausible explanation for poor P and K transfer. It is also listed as Limitation vii.

*Reference precision and attainable R² ceiling for Olsen P.*

**Response:** The deposit does not report replicate precision, so we give a sensitivity analysis (Section 3.9). Even with a 30% relative error plus a 1 mg/kg floor, the attainable R² would be about 0.90, far above what any model achieved. The P failure is therefore not explained by reference precision alone, and the phrase "intrinsically untransferable" has been removed.

*Preprocessing choices not explored.*

**Response:** Five alternatives were tested (Table S5): no CO₂ mask, SNV before the derivative, SNV only, a second derivative, and coarser binning. The adopted pipeline was within 0.04 R² of the best alternative. SNV-only gave slightly higher values for KRR but much lower values for PLSR.

*A random-split reference.*

**Response:** Added as a fifth design (Section 3.1, Table 3, Figure 3). Moving from random splitting to leave-site-out, leave-AEZ-out and leave-country-out removed 32%, 39% and 41% of the apparent skill. For Olsen P, R² fell from 0.52 under random splitting to at most 0.08 for a new site.

*Two depths per location — make explicit that grouping handles this.*

**Response:** Stated explicitly in Section 2.7. Both depths of a profile share a cluster and a site, so every design keeps them together. In the spiking experiment, profile partners of spiked samples were also excluded from the test set.

**8. References and their uses (reviewer section 4)**

*Benchmarks missing: Cubist, memory-based learning, SVR/KRR/GPR, a plain 1D-CNN and a multi-output CNN.*

**Response:** All six were added (Section 2.5), so there are now twelve benchmarks.

*OSSL, global libraries, calibration transfer and spiking.*

**Response:** We now cite Viscarra Rossel *et al.* (2016), Safanelli *et al.* (2025), Guerrero *et al.* (2010), Wetterlind and Stenberg (2010) and Gamagedara *et al.* (2025, corrected). Spiking is now tested experimentally (Section 3.6, Table 8, Figure 8).

*Unlabelled spectra / self-supervised pretraining.*

**Response:** We agree that this is the natural next step. We did not attempt it in this revision because a fair test would require the same nested, multi-seed protocol for the pretraining stage, which would double the neural compute. It is listed explicitly as Limitation v and as a priority in the Conclusion.

*Spatial validation literature: Meyer et al., Wadoux et al. (2021), Valavi (blockCV), Milà et al. (2022).*

**Response:** All are now cited and discussed in the Introduction, together with Meyer *et al.* (2018), Meyer and Pebesma (2021) and Linnenbrink *et al.* (2024). The area-of-applicability concept is also used when interpreting the conformal results (Section 3.7).

**9. Experimental improvements requested**

> 1\. **Local spiking (k = 0, 5, 10, 20, 50):** done (Section 3.6, Table 8, Figure 8). Mean R² for kernel ridge regression rose from 0.512 to 0.658. Olsen P improved only to about 0.06.
>
> 2\. **Learning curve over samples and sites:** done (Section 3.5, Figure 7, Table S8).
>
> 3\. **Ablation table:** done (Table 5, Figure 6).
>
> 4\. **Direct test of boundedness with ‖W~o~‖₁ on log and original scales:** done (Table 6, Section 3.4).
>
> 5\. **Group-bootstrap CIs and paired tests:** done (Sections 2.8 and 3.2, Tables 4 and S3).
>
> 6\. **Multiple seeds (5–10):** done; five seeds for all neural models (Section 2.6).
>
> 7\. **Leave-AEZ-out and group-aware conformal prediction:** done. Leave-AEZ-out uses five Köppen–Geiger groups (Section 2.1, Table 3). With group-aware calibration, conformal coverage of nominal 90% intervals was 84.5% under leave-site-out, 85.4% under leave-AEZ-out and 80.5% under leave-country-out, against 71.7–78.5% with random calibration (Section 3.7, Table 9, Figure 9).

We thank the reviewer again. The report has substantially improved the rigour and honesty of the paper.
