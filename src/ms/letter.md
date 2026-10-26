**Cover Letter and Point-by-Point Response to the Reviewer**

**Manuscript ID:** 1283

**Title:** "Comparison of a Hybrid Deep Learning Model with Machine Learning Methods for Mid-Infrared Soil Nutrient Prediction under Spatial Transfer in East Africa"

**Running title:** Spectral soil nutrient prediction under spatial transfer

**Journal:** Journal of Agriculture, Science and Technology (JAGST)

**Decision:** Revisions Required

Dear Editor,

Thank you for the opportunity to revise our manuscript, and thank you to the reviewer for a careful and constructive report. The reviewer asked for stronger evidence on the architecture, the validation design, the statistics and the agronomic interpretation, and proposed seven specific experiments. We carried out all seven, re-ran every analysis with nested, group-aware tuning and multiple random seeds, and rewrote the manuscript around the results. The revised manuscript is a self-contained report of this analysis.

**Response to the editor's instructions**

> \- **(I) References.** Every reference was checked against the original source, is relevant to the text, and carries a DOI or URL. Two bibliographic records were corrected (Gamagedara *et al.*, 2025; Yin *et al.*, 2026), and references requested by the reviewer were added. Every citation is listed and every listed reference is cited.
>
> \- **(II) Track changes.** Two versions are uploaded: one with all changes tracked and one clean version. Because the analysis was redone, most of the text appears as tracked changes.
>
> \- **(III–IV) Point-by-point responses.** These follow below. We were able to address every comment.
>
> \- **(VI) English.** The whole manuscript was rewritten and edited for clarity, consistent terminology and concise style.
>
> \- **(IX) Anonymity.** Author names, affiliations, the corresponding e-mail address, institutional acknowledgements and the repository link have been removed from all files and from the file metadata.

**Summary of the seven experiments requested by the reviewer**

| No. | Reviewer request | What we did | Main result | Where |
|---|---|---|---|---|
| 1 | Local spiking, k = 0, 5, 10, 20, 50 | Added k samples from the held-out site to training under leave-site-out; profile partners of spiked samples excluded from the test set | Mean R² of kernel ridge regression rose from 0.512 (k = 0) to 0.658 (k = 50); Olsen P stayed near 0.06 | Section 3.6, Table 8, Fig. 8 |
| 2 | Learning curve over samples and sites | Crossed training size n = 100–800 with the number of training sites m = 2 to all, plus a like-for-like test on the Kenyan sites | At n = 200, R² rose from 0.04 (2 sites) to 0.47 (all sites); quadrupling n added only 0.06 | Section 3.5, Table 7, Fig. 7, Table S7 |
| 3 | Ablation table | Nine single-component replacements of the hybrid under leave-site-out, all 13 properties | The RBF head and the gate gave no benefit; positional information and attention pooling were essential | Section 3.3, Table 5, Fig. 6 |
| 4 | Direct boundedness test, ‖W~o~‖₁, log and original scales | Hybrid compared with an otherwise identical linear-head network; ‖w~o~‖₁ reported | Median ‖w~o~‖₁ = 5.49, a loose bound; worst-case R² similar on both scales | Section 3.4, Table 6 |
| 5 | Group-bootstrap CIs and paired tests | 2,000-draw group bootstrap over spatial units; paired tests for all model pairs; Friedman and Nemenyi tests | GPR ranked first; eleven benchmarks were within the critical difference; only the hybrid fell outside it | Sections 2.8 and 3.2, Table 4, Table S3 |
| 6 | 5–10 seeds for all neural models | Five seeds (11, 22, 33, 44, 55) for the MLP, CNN, multi-output CNN and hybrid in every fold | The hybrid had the largest seed variability (SD of R² 0.041) | Section 2.6 |
| 7 | Leave-AEZ-out and group-aware conformal prediction | Five Köppen–Geiger zones withheld in turn; split-conformal intervals with a group-aware calibration set | Coverage of nominal 90% intervals 84.5% (site), 85.4% (AEZ) and 80.5% (country), against 71.7–78.5% with random calibration | Sections 2.1 and 3.7, Tables 3 and 9, Fig. 9 |

**Title and format**

The title was changed to "Comparison of a Hybrid Deep Learning Model with Machine Learning Methods for Mid-Infrared Soil Nutrient Prediction under Spatial Transfer in East Africa". It states the comparison and the validation setting that the paper reports, follows the descriptive style of comparison studies published in JAGST, and avoids abbreviations. The phrase "multi-modal fusion" was removed, as the reviewer suggested, because the ancillary input is limited to sampling depth.

The manuscript follows the JAGST Author Guidelines and submission checklist: Calibri 11 pt, single line spacing; an abstract of fewer than 300 words with all abbreviations defined; a running title of seven words at the foot of the title page; Harvard in-text citations, with multiple citations in alphabetical order separated by semicolons; an alphabetical reference list starting on a new page, with a link for every reference; tables and figures placed in the text where they are first discussed; metric units; and an Ethical Statement (Section 6.0). The text is original and was written for this manuscript.

We hope the revised manuscript now meets the standard of JAGST, and we would be glad to make any further changes.

Yours sincerely,

The Authors

**Point-by-point response**

Reviewer comments are in *italics*, and our responses follow. Section, table and figure numbers refer to the revised manuscript.

***General comment.*** *The great contribution of the paper is the nested group-disjoint validation … the claims (calibration breadth beats architecture, boundedness explains robustness, attention and gate are interpretable, widespread P deficiency) each have a confound or a validity problem.*

**Response:** We agree, and each claim was tested directly:

> \- Breadth versus architecture: tested with a like-for-like design and learning curves (Section 3.5).
>
> \- Boundedness: tested directly; the bounded head gave no practical protection, and the paper says so (Section 3.4).
>
> \- Interpretability: attention is validated against occlusion, integrated gradients and PLSR VIP, and the gate is not interpreted as a contribution (Section 3.8).
>
> \- P deficiency prevalence: replaced by screening accuracy against critical levels, with no population prevalence claimed (Section 3.9).

Nested validation is now the centre of the paper and covers five designs: random, cluster, site, agro-ecological zone and country.

**1. Architecture claims (reviewer section 2.1)**

*1.1 No ablations: (a) linear or MLP head, (b) attention versus mean or max pooling, (c) gate versus concatenation, (d) spectra-only versus spectra plus ancillary, (e) RBF head versus dense head.*

**Response:** Done (Sections 2.9 and 3.3, Table 5, Fig. 6). Nine single-component replacements were run under leave-site-out with three seeds each (five for the linear head), over all 8 folds and 13 properties:

> \- (a/e) Linear head: ΔR² = +0.015. Dense ReLU head: +0.016. Fixed-centre Gaussian head: −0.010. The RBF head therefore gives no benefit.
>
> \- (b) Mean pooling: −0.217, harming all 13 properties. Mean pooling was used as the comparator because, like attention, it pools over positions. Attention re-weighting followed by flattening gave +0.008, and removing the positional embedding gave −0.192.
>
> \- (c) Concatenation instead of the gate: +0.013. The gate therefore gives no benefit.
>
> \- (d) Spectra only: −0.039, with significant losses for 5 properties.
>
> \- Adding coordinates: −0.011.

The manuscript states plainly that the architecture does not achieve its design objective (Sections 3.3 and 4.0).

*1.2 The "RBF head" is not a kernel method … Intro gap 1 is a straw man … cite deep kernel learning, differentiable GP/SVR layers and RBF layers.*

**Response:** Agreed. The first research question was rewritten (Section 1.0). Joint training of RBF and kernel components with neural encoders is presented as well established (Park and Sandberg, 1991; van Amersfoort *et al.*, 2020; Wilson *et al.*, 2016), and the question posed is whether such a head *improves* prediction, which only ablation can answer. The head is a genuine RBF layer with learned centres, φᵢ = exp(−γᵢ‖u − cᵢ‖²) (Eq. 3, Section 2.4.4), and the text states that it is not a kernel machine or a support vector regressor.

*1.3 The propositions are trivial … they do not justify "the architecture cannot fail without bound".*

**Response:** Agreed. The propositions were removed. The manuscript gives only the one-line bound |ŷ − b~o~| ≤ ‖w~o~‖₁ as a property of the construction, states that it is not evidence of robustness, and tests it empirically (Section 3.4).

*1.4 Single seed for neural nets.*

**Response:** Every neural model (MLP, CNN, multi-output CNN and hybrid) was trained from **five random initialisations** in every fold of every design. Results are seed means with the seed standard deviation, and seed variability is carried into all bootstrap intervals (Section 2.6). The hybrid had the largest seed variability (mean SD of R² 0.041, against 0.028 for the CNN and 0.022 for the MLP).

*1.5 Weak tuning; equal budget is not equal fairness.*

**Response:** Tuning is nested inside each outer training fold with group-disjoint inner splits (Section 2.6), and search spaces are sized to each model (Table S2): 7 values for PLSR, 11 for ridge, two-dimensional grids for SVR and KRR, marginal-likelihood optimisation for GPR, early-stopped tree numbers for LightGBM, and three architectures per neural model, each trained with five seeds. The neural search space is modest because of the seed and fold multiplicity, and this is stated as a limitation (Section 3.12, item vi). The plain CNN with the same budget performed on a par with the best conventional models, so the hybrid's ranking is not explained by under-tuning; the ablation identifies the components responsible.

*1.6 Early-stopping leakage.*

**Response:** Agreed and corrected. Early stopping and configuration choice use a **group-disjoint inner split** inside the training fold: whole clusters for the cluster design and whole sites otherwise (Section 2.6). The outer test fold is never used for any decision, and all results were regenerated under this protocol.

*1.7 Self-implemented baselines. Use scikit-learn, LightGBM/XGBoost and PyTorch, and report agreement.*

**Response:** All models now use maintained libraries: scikit-learn (PLSR, ridge, SVR, KRR, GPR, random forest), LightGBM, the cubist package, and PyTorch for the neural models. Memory-based learning, which has no maintained Python package, is built from scikit-learn components following Ramirez-Lopez *et al.* (2013). Because every result in the revised manuscript comes from these library implementations, we report the agreement check here rather than in the paper. Under the original protocol, ridge regression reproduced exactly, and random forest and gradient boosting agreed closely (median absolute difference in fold R² of 0.023 and 0.022). PLSR differed (0.147); the check showed that our own PLSR code had computed its coefficients without the weight matrix. This is why all models were moved to library implementations, and the PLSR results in the paper are from scikit-learn.

**2. Calibration breadth (reviewer section 2.2)**

*Kenya-only versus four-country changes sample size, number of sites and diversity at once; test populations differ … add a learning curve varying n and number of sites, and the like-for-like comparison … the "order of magnitude" statement is not supported.*

**Response:** Both experiments were added (Section 3.5, Table 7, Fig. 7, Table S7).

> \- **Like-for-like comparison.** The test set is always the three Kenyan sites. Mean R² was −0.136 when training on the other two Kenyan sites, 0.101 with a size-matched set drawn from about 13 regional sites, 0.108 with non-Kenyan data only, and 0.230 with Kenyan plus regional data. At constant size, diversity improved 9 of 13 properties.
>
> \- **Learning curves.** n = 100–800 was crossed with m = 2 to all training sites. For kernel ridge regression at n = 200, mean R² rose from 0.04 with two sites to 0.47 with all sites, whereas quadrupling n with all sites added only 0.06.

The "order of magnitude" statement was removed. The breadth effect is compared with the full spread among the twelve benchmarks (0.057 under leave-site-out), and the conclusion is stated cautiously.

**3. Boundedness (reviewer section 2.3)**

*The boundedness argument is contradicted by your own table … a likely confound is the log1p back-transform … report ‖W_o‖₁ and test the claim directly … saturating to zero is not abstention.*

**Response:** We agree. The test is reported in Section 3.4 and Table 6, and the boundedness argument is not made in the revised paper.

> \- The learned ‖w~o~‖₁ had a median of 5.49 (5th–95th percentile 3.23–7.96, maximum 10.7) in standardised units, so the bound is loose.
>
> \- Against an otherwise identical linear-head network, worst-case R² was similar on both the original and the log scale, under leave-site-out, leave-AEZ-out and leave-country-out.
>
> \- The reviewer's explanation is confirmed. Without a range safeguard, five model–property combinations had R² below −1, all for total S, which is log-modelled and has skewness 13.9. They involved ridge, kernel and GP models and the MLP, and none involved the hybrid.
>
> \- Limiting back-transformed predictions to the training range removed these failures. This safeguard is applied identically to all models and is reported in Section 2.8.

The manuscript notes that an RBF head whose units output values near zero returns a bias-driven prediction without an uncertainty signal; group-aware conformal intervals provide that signal (Section 3.7).

**4. Interpretability (reviewer section 2.4)**

*Attention is not explanation … validate with occlusion/perturbation, integrated gradients, PLSR VIP and a permutation test.*

**Response:** Done (Section 3.8, Fig. 10, Table S5). Agreement of attention with the other measures (Spearman ρ):

> \- integrated gradients: 0.82
>
> \- occlusion: 0.39
>
> \- PLSR VIP: 0.16

A permutation test with circular shifts, which preserves autocorrelation, showed attention enrichment inside literature absorbance windows of 1.6–1.8 times the null for organic C, total N and pH (combined p = 0.02–0.04), although few single-fold tests were significant. Attention is therefore described as a coarse, hypothesis-generating summary (Jain and Wallace, 2019; Wiegreffe and Pinter, 2019).

*Resolution: quoting single wavenumbers overclaims precision.*

**Response:** Agreed. The receptive field of each attended position is reported as about 286–417 cm⁻¹, with positions about 31 cm⁻¹ apart (Section 2.4.1), and attention is described only in terms of broad regions.

*Band assignments; 2454 cm⁻¹ is near the CO₂ region; mask atmospheric regions.*

**Response:** Agreed. The CO₂ region (2450–2250 cm⁻¹) is removed before modelling (Section 2.3), and band assignments cite Janik *et al.* (1998), Nguyen *et al.* (1991) and Parikh *et al.* (2014). A preprocessing sensitivity analysis (Table S4) shows that masking changes transfer skill by less than 0.01 R².

*The gate is not an identifiable contribution.*

**Response:** Agreed. Gate weights are reported only descriptively, and contributions are estimated by permutation and by the spectra-only ablation (Section 3.8). For Olsen P and exchangeable K, depth was nearly as informative as the spectra; for organic C and pH it contributed little.

*Ancillary branch uses coordinates … do benchmarks receive the same features? "Multi-modal fusion" overstates.*

**Response:** All models receive **identical inputs**: the spectrum plus depth (Section 2.5). Coordinates are excluded from the main analysis because of the extrapolation hazard (Meyer and Pebesma, 2021) and are tested as an ablation; they did not help the hybrid (ΔR² −0.011) or LightGBM (below 0.001 in magnitude). "Multi-modal fusion" was removed from the title, and the text uses "spectral–ancillary fusion".

**5. Agronomic section (reviewer section 2.5)**

*Total Cu/Zn thresholds are not meaningful; headline percentages are invalid; non-probability sample; RQ4 needs yield data.*

**Response:** Agreed on all points.

> \- No deficiency is assessed for total Cu, Zn, quasi-total S or total N, with the reason given (DTPA extraction; Lindsay and Norvell, 1978).
>
> \- No population prevalence is claimed for any property.
>
> \- The fourth research question asks which properties with established interpretive classes can be **screened** in an unsampled landscape.
>
> \- Table 10 reports sensitivity, specificity and balanced accuracy for pH, organic C, Olsen P, exchangeable Ca, Mg and K, and eCEC against indicative critical levels (Landon, 1991). Screening was reliable for organic C (balanced accuracy 0.91), pH (0.84) and eCEC (0.83), and close to chance for Olsen P (0.58).

**6. Statistics and metrics (reviewer section 2.6)**

*Per-fold R² unstable; add RMSE/SD, RPD, RPIQ, Lin's CCC and MAE relative to a mean baseline.*

**Response:** Metrics are computed on pooled out-of-fold predictions. We report R², RMSE, RMSE/SD~train~, RPD, RPIQ, CCC, MAE skill against the training-mean baseline, and log-scale R² (Section 2.8, Table 4, Table S3). Per-fold R² is reported only as a range.

*CIs not valid (t-intervals over few folds; interval above 1).*

**Response:** Intervals come from a **group bootstrap** (2,000 draws) over the spatial units, averaged over seeds (Section 2.8).

*"Best R²" selected post hoc.*

**Response:** Model-level comparisons use GPR, the model with the best mean rank. The per-property maximum appears only in a separate column marked as descriptive and optimistic (Table 4).

*No significance tests.*

**Response:** Paired group-bootstrap tests for all model pairs, the Friedman test and the Nemenyi critical difference were added (Section 3.2). All eleven other benchmarks were within the critical difference of GPR; only the hybrid fell outside it. No conclusion rests on a single fold.

*Folds actually evaluated; Ethiopia never tested; state site grouping.*

**Response:** With nested tuning, every fold serves once as the test fold in every design, so the country design tests all four countries, including Ethiopia. Table S1 lists the sites in every leave-site-out fold and the composition of all other folds.

*Tuning leakage (single fold); use nested group CV.*

**Response:** Done (see 1.5).

**7. Data and design (reviewer section 3)**

*Time gap between spectra (2009–2013) and chemistry (2017–2018).*

**Response:** Discussed in Sections 2.2 and 3.9, citing Bartlett and James (1980), as a plausible explanation for poor P and K transfer, and listed as a limitation (Section 3.12, item vii).

*Reference precision and attainable R² ceiling for Olsen P.*

**Response:** The deposit does not report replicate precision, so a sensitivity analysis is given (Section 3.9). Even with a 30% relative error plus a 1 mg/kg floor, the attainable R² would be about 0.90, far above what any model achieved. Reference precision alone therefore does not explain the P result.

*Preprocessing choices not explored.*

**Response:** Five alternatives were tested (Table S4): no CO₂ mask, SNV before the derivative, SNV only, a second derivative, and coarser binning. The adopted pipeline was within 0.04 R² of the best alternative.

*A random-split reference.*

**Response:** Added as a fifth design (Section 3.1, Table 3, Fig. 3). Moving from random splitting to leave-site-out, leave-AEZ-out and leave-country-out removed 32%, 39% and 41% of the apparent skill. For Olsen P, R² fell from 0.52 under random splitting to at most 0.08 for a new site.

*Two depths per location — make explicit that grouping handles this.*

**Response:** Stated in Section 2.7: both depths of a profile share a cluster and a site, so every design keeps them together. In the spiking experiment, profile partners of spiked samples were excluded from the test set (Section 2.9).

**8. References and their uses (reviewer section 4)**

*Benchmarks missing: Cubist, memory-based learning, SVR/KRR/GPR, a plain 1D-CNN and a multi-output CNN.*

**Response:** All six were added (Section 2.5), giving twelve benchmarks.

*OSSL, global libraries, calibration transfer and spiking.*

**Response:** Cited: Viscarra Rossel *et al.* (2016), Safanelli *et al.* (2025), Guerrero *et al.* (2010), Wetterlind and Stenberg (2010) and Gamagedara *et al.* (2025). Spiking is tested experimentally (Section 3.6, Table 8, Fig. 8).

*Unlabelled spectra / self-supervised pretraining.*

**Response:** We agree that this is the natural next step. A fair test would need the same nested, multi-seed protocol for the pretraining stage, roughly doubling the neural computation, so it is listed as a limitation (Section 3.12, item v) and as a priority in the Conclusion.

*Spatial validation literature: Meyer et al., Wadoux et al. (2021), Valavi (blockCV), Milà et al. (2022).*

**Response:** All are cited and discussed in the Introduction, together with Meyer *et al.* (2018), Meyer and Pebesma (2021) and Linnenbrink *et al.* (2024). The area-of-applicability concept is used when interpreting the conformal results (Section 3.7).

We thank the reviewer again. The report has substantially improved the rigour of the paper.
