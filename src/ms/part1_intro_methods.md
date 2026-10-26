**1.0 Introduction**

Agriculture underpins the food security and economic stability of most households in sub-Saharan Africa. Despite this central role, smallholder farming systems consistently deliver yields well below their biophysical potential, and soil nutrient depletion is among the most persistent causes. Fertiliser recommendations across much of East Africa are still issued as blanket rates for large administrative units, even though soil fertility varies substantially over distances of a few hundred metres. Closing this gap requires soil information at a spatial density that conventional wet-chemistry analysis cannot supply affordably.

Mid-infrared (MIR) diffuse reflectance spectroscopy offers a credible route past this constraint. A soil spectrum records the vibrational absorbance of mineral and organic constituents, and one measurement taking less than a minute can support the simultaneous estimation of several properties (Janik *et al.*, 1998; Soriano-Disla *et al.*, 2014). The technique has been developed for tropical soils over two decades (Shepherd and Walsh, 2002), and the Africa Soil Information Service (AfSIS) has assembled spectral libraries at continental scale (Hengl *et al.*, 2021; Vågen *et al.*, 2016). Global and open libraries now extend this foundation further (Safanelli *et al.*, 2025; Viscarra Rossel *et al.*, 2016).

Turning a spectral library into reliable agronomic information is a modelling problem. Partial least squares regression (PLSR) remains the operational default. Rule-based Cubist models (Minasny and McBratney, 2008), memory-based learning with local PLS (Ramirez-Lopez *et al.*, 2013) and kernel methods are established alternatives. Convolutional neural networks (CNNs), including multi-output CNNs that predict several properties at once, have been reported to improve on PLSR, particularly with large libraries (Ng *et al.*, 2019; Padarian *et al.*, 2019; Shen and Viscarra Rossel, 2021). More recently, attention-based designs have appeared: Guo *et al.* (2025) transform MIR spectra into two-dimensional images for a Swin Transformer, Shuai *et al.* (2025) couple channel attention with a ConvNeXt backbone for visible–near-infrared data, and Yin *et al.* (2026) use self-attention to weight an ensemble of spectral learners. Using the same AfSIS Phase I archive as the present study, Gruszczyński and Gruszczyński (2022) found that no single algorithm was optimal for all soil variables, and that available phosphorus and several micronutrients were poorly predicted.

Three questions about this literature motivate the present study.

The first concerns **what radial basis function (RBF) components add to a neural network**. RBF networks with learned centres are universal approximators (Park and Sandberg, 1991), and RBF output layers have been combined with deep feature extractors to give distance-aware predictions (van Amersfoort *et al.*, 2020). Kernel methods can also be trained jointly with neural encoders, as in deep kernel learning (Wilson *et al.*, 2016). The question is therefore not whether such a head *can* be trained end to end, which is well established, but whether it *improves* soil spectral prediction relative to a linear or dense head on the same encoder. This can only be answered by ablation, which most architectural studies in soil spectroscopy do not report.

The second concerns **whether attention weights explain a prediction**. Attention weights are often presented as making a network's reasoning transparent. In natural language processing it has been shown that attention distributions need not agree with other importance measures and can be changed without changing the prediction (Jain and Wallace, 2019; Wiegreffe and Pinter, 2019). For spectra, an attention weight has a physical referent (a wavenumber window), but that window can be wide after convolution and pooling. Attention therefore needs to be checked against perturbation-based attribution and known absorbance bands before it is interpreted.

The third concerns **validation design and spatial transfer**, that is, prediction at sites, climatic zones or countries that contributed no samples to calibration. Soil spectral libraries are assembled from clustered sampling campaigns. Random splitting places near-duplicate samples on both sides of the partition, so the resulting accuracy describes interpolation among neighbours rather than prediction at a new location (Ploton *et al.*, 2020; Roberts *et al.*, 2017). Spatial and target-oriented cross-validation (Meyer *et al.*, 2018; Valavi *et al.*, 2019), the area of applicability (Meyer and Pebesma, 2021) and prediction-domain-matched designs (Linnenbrink *et al.*, 2024; Milà *et al.*, 2022) have been proposed in response, while Wadoux *et al.* (2021) caution that spatial cross-validation estimates a specific extrapolation task rather than map accuracy in general. For spectral libraries the corresponding practical questions are how much skill survives transfer to a new landscape or country, whether transfer is limited by the size or by the diversity of the calibration set, and how much a few local samples recover. Spiking (Guerrero *et al.*, 2010; Wetterlind and Stenberg, 2010) and calibration transfer (Gamagedara *et al.*, 2025) are the established responses to that last question.

This study addresses these three questions by comparing a hybrid deep learning model, an attention-weighted CNN–RBF network, with twelve established machine learning methods under spatial transfer in East Africa, using an evaluation designed to test, rather than assume, the contribution of each part of the hybrid. Specifically, we ask:

> 1\. Under nested, group-aware tuning and multiple random initialisations, does the hybrid model, or any of its components (attention, modality gate, RBF head), improve soil nutrient prediction over twelve established machine learning benchmarks?
>
> 2\. How does predictive skill change as the withheld unit widens from a random sample to a sampling cluster, a sentinel site, a climatic agro-ecological zone and a country, and is transfer limited by calibration size or by calibration diversity?
>
> 3\. How much skill do a few local samples recover in a new landscape, and can group-aware conformal prediction provide intervals that remain valid under transfer?
>
> 4\. Do attention weights agree with perturbation-based attribution and with known MIR absorbance regions, and which agronomically interpretable properties can be screened reliably in an unsampled landscape?

We report each outcome as the evidence supports it, including where the results do not favour the hybrid.

**2.0 Materials and Methods**

**2.1 Study area and sampling design**

Samples were drawn from 16 AfSIS Land Degradation Surveillance Framework (LDSF) sentinel sites in four East African countries (Table 1, Figure 1). Each site is a 10 × 10 km block sampled on a randomised, spatially stratified hierarchical design in which clusters are placed within the block and plots within clusters (Vågen *et al.*, 2010). Kenya contributes three sites spanning a pronounced agro-ecological gradient: Morijo in the highland Rift Valley of Narok County, Mbalambala in the arid eastern lowlands of Garissa County, and Marafa in the coastal hinterland of Kilifi County. Ethiopia contributes four sites, Tanzania seven and Uganda two. Three Tanzanian sites (Bukwaya, Kiberashi and Mbinga) contribute only [[n_tiny]] samples between them; they are retained, but they cannot support site-level statistics on their own. Samples were taken at two depths, topsoil (0–20 cm, n = 659) and subsoil (20–50 cm, n = 651), from the same plots, so each topsoil sample has a subsoil partner from the same profile.

For the agro-ecological analysis each site was assigned the Köppen–Geiger climate class (Kottek *et al.*, 2006) of the majority of its sample locations: tropical savanna (Aw; [[aez_n_Aw]] samples, five sites), tropical monsoon (Am; [[aez_n_Am]], one site), hot semi-arid steppe (BSh; [[aez_n_BSh]], four sites), hot desert (BWh; [[aez_n_BWh]], one site) and temperate highland climates (Cfb, Cwb, Csb and Cwa, combined as C; [[aez_n_C]], five sites). Climate class is used here as a transparent, reproducible proxy for agro-ecological zone; it is not a substitute for a national agro-ecological zonation.

Table 1: Composition of the analysis set. Samples are AfSIS Land Degradation Surveillance Framework records with paired MIR spectra and wet chemistry, collected in 2009–2013 at two depths. Köppen–Geiger groups as defined in Section 2.1.

[[TABLE:T1]]

![](FIG1)

Fig. 1. Study area and sampling structure. (a) The 16 sentinel sites, coloured by Köppen–Geiger group, with marker shape showing country and marker size proportional to the number of samples. (b) Samples per site. (c) Mean SNV-corrected MIR spectrum for each Köppen–Geiger group; the atmospheric CO₂ region removed before modelling is shaded.

The 1,310 samples are a stratified sample of landscapes rather than a probability sample of East African cropland, so the data support statements about predictive performance in these landscapes but not population-level prevalence estimates.

**2.2 Data sources and dataset assembly**

Two openly licensed AfSIS deposits were joined on the soil sample number. Vågen *et al.* (2020) provide MIR spectra for about 18,500 AfSIS Phase I samples, measured on a Bruker Tensor 27/HTS-XT FT-IR spectrometer at the World Agroforestry (ICRAF) Soil–Plant Spectral Diagnostics Laboratory in Nairobi. The Kenya, Ethiopia, Tanzania and Uganda files hold 4,654 spectra across 1,749 wavenumbers from 601.7 to 4001.6 cm⁻¹. Vågen *et al.* (2021) provide wet chemistry for 2,002 archived Phase I samples analysed at Rothamsted Research in 2017–2018, of which 1,310 belong to the four study countries. The assembly script verifies that every country file shares an identical wavenumber grid, and the join was complete: all 1,310 chemistry records matched a spectrum.

Thirteen properties were modelled (Table 2): pH in water (1:2.5); organic C and total N by dry combustion; Olsen-extractable P; exchangeable Ca, Mg and K and effective cation exchange capacity (eCEC) by cobalt hexamine extraction with inductively coupled plasma optical emission spectrometry (ICP-OES); and quasi-total S, Fe, Mn, Zn and Cu by aqua regia digestion with ICP-OES. Missing values range from none to four per property. The deposit's variable dictionary transposes the descriptions of the total N and total C columns; the column names are correct, as the observed medians confirm, and the names were used.

Table 2: Soil properties modelled. Skew is computed on all samples; the log(1 + y) decision was made within each training fold. The last column marks properties whose analytical method measures a fraction for which agronomic critical levels are commonly used (Section 3.9).

[[TABLE:T2]]

Two features of the data matter for interpretation. First, the spectra were acquired on samples collected in 2009–2013, while the reference chemistry was determined on the archived material in 2017–2018. Storage of air-dried soil can change extractable P, exchangeable K and mineral N (Bartlett and James, 1980), so part of the error for these properties may reflect a mismatch between the sample that was scanned and the sample that was analysed, rather than a spectral limitation. Second, Olsen P is strongly skewed (median [[p_median]] mg/kg; [[p_below2:.0f]]% of samples below 2 mg/kg), so many values lie close to the range in which analytical precision is poorest. The deposit does not report replicate precision, so the attainable R² for Olsen P cannot be estimated directly; Section 3.9 gives a sensitivity analysis.

**2.3 Spectral preprocessing**

Spectra were smoothed and differentiated with a Savitzky–Golay filter (window 11 points, second-order polynomial, first derivative), corrected for multiplicative scatter with the standard normal variate (SNV), and averaged in bins of four adjacent wavenumbers. The atmospheric CO₂ region (2450–2250 cm⁻¹) was removed before binning, leaving [[n_bins]] bins of about 7.8 cm⁻¹. These operations act on each spectrum separately and share no statistic between samples. The only data-dependent step, per-bin standardisation, was fitted on training rows only. Target variables whose training-fold skewness exceeded 2.0 were modelled on the log(1 + y) scale; predictions were back-transformed before any metric was computed, so all errors are in the original units. The preprocessing choices were tested against five alternatives (no CO₂ mask; SNV applied before the derivative; SNV only; a second derivative; bins of eight) in a sensitivity analysis with PLSR and kernel ridge regression under leave-site-out (Table S4).

**2.4 Hybrid deep learning model: attention-weighted CNN–DAFM–RBF network**

The hybrid model (Figure 2; referred to below as the hybrid) has a spectral branch, an ancillary branch, a modality gate and an RBF regression head, trained end to end.

***2.4.1 Spectral branch.*** Two one-dimensional convolutional layers with ReLU activation and an intervening max-pooling layer produce a feature map **H** ∈ ℝ^p×C^, where p indexes spectral positions and C the learned channels:

EQ:: *h*~j,l~ = σ(Σ~k~ *w*~k,l~ *x*~j+k−1~ + *b*~l~)   (1)

A learned positional embedding **e**ⱼ ∈ ℝ^C^ is added to each position (**h**ⱼ ← **h**ⱼ + **e**ⱼ). Without it, any pooling over positions is permutation-invariant and cannot tell where in the spectrum a feature occurred; its contribution is tested as an ablation. With the selected configuration, each position integrates a receptive field of [[rf_bins]] input bins (about [[rf_cm:.0f]] cm⁻¹), and adjacent positions are [[jump_cm:.0f]] cm⁻¹ apart. Attention weights therefore refer to broad spectral windows, not to individual wavenumbers.

![](FIG2)

Fig. 2. The hybrid attention-weighted CNN–DAFM–RBF architecture. The spectral branch applies two convolutional layers with a learned positional embedding; the Dynamic Attention Fusion Module pools positions with additive attention; a softmax gate combines the spectral context with the encoded ancillary input (depth); and an RBF layer with learned centres produces the prediction, which is back-transformed for log-modelled properties. The components replaced in the ablation study are listed at upper right.

***2.4.2 Dynamic Attention Fusion Module (DAFM).*** Additive attention over positions forms a spectral context vector:

EQ:: *e*~j~ = **v**^⊤^ tanh(**W** **h**~j~ + **b**),  α~j~ = exp(*e*~j~) ⁄ Σ~k~ exp(*e*~k~),  **z**~s~ = Σ~j~ α~j~ **h**~j~   (2)

***2.4.3 Ancillary branch and modality gate.*** A dense layer encodes the ancillary input, sampling depth, into **z**~c~. A softmax gate weights the two branches for each sample, **g** = softmax(**W**~g~[**z**~s~; **z**~c~] + **b**~g~), and the fused vector is **z** = [g₁**z**~s~; g₂**z**~c~]. Because **g** multiplies vectors of different dimension and scale, and the encoders can rescale to compensate, g₁ is not an identifiable share of the prediction. It is reported only descriptively; modality contributions are estimated instead by ablation and permutation (Section 2.9).

***2.4.4 RBF head with learned centres.*** The fused vector is projected to a low-dimensional space, **u** = **W**~r~**z** + **b**~r~, and compared with R learned centres **c**ᵢ:

EQ:: φ~i~ = exp(−γ~i~ ‖**u** − **c**~i~‖²),  *ŷ* = **w**~o~^⊤^ **φ** + *b*~o~   (3)

with γᵢ = exp(log γᵢ) > 0. This is a standard RBF network layer with learned centres (Park and Sandberg, 1991) placed on a neural encoder, in the spirit of van Amersfoort *et al.* (2020). It is not a support vector regressor or a kernel machine. A simpler variant, a dense layer followed by a Gaussian activation centred at zero (φᵢ = exp(−γᵢpᵢ²)), is tested as an ablation. Because 0 < φᵢ ≤ 1, the output satisfies |ŷ − b~o~| ≤ ‖**w**~o~‖₁. This bound is a property of the construction, not evidence of robustness: ‖**w**~o~‖₁ is learned and can be large, and for log-modelled targets a bounded output on the log scale can still be large after back-transformation. Whether the bound limits failures in practice is therefore tested directly (Section 3.4), and the learned ‖**w**~o~‖₁ is reported.

**2.5 Machine learning benchmark models**

Twelve benchmarks were run: PLSR, ridge regression, support vector regression (SVR) with an RBF kernel, kernel ridge regression (KRR) with an RBF kernel, Gaussian process regression (GPR) with an RBF plus white-noise kernel and marginal-likelihood hyperparameters, and random forest, all from scikit-learn (Pedregosa *et al.*, 2011); gradient boosting with LightGBM; Cubist (regression rules with committees and optional instance-based correction, on the first 30 principal components; Minasny and McBratney, 2008); memory-based learning (MBL), in which each prediction is a weighted average of local PLS models fitted to the k nearest training spectra in whitened principal-component space (Ramirez-Lopez *et al.*, 2013); a multilayer perceptron (MLP); a plain 1D-CNN with flattened convolutional features and a dense head (Padarian *et al.*, 2019); and a multi-output version of that CNN predicting all 13 properties jointly with a masked loss (Ng *et al.*, 2019). The neural networks were implemented in PyTorch. **Every model received the same inputs**: the preprocessed spectrum and the depth flag. Coordinates were excluded from all main analyses because latitude and longitude are a known hazard under spatial extrapolation (Meyer and Pebesma, 2021); their effect is tested in an ablation. Because the ancillary input is limited to depth, we refer to the combination as spectral–ancillary fusion.

All models except MBL were fitted with maintained, widely used libraries, so that the benchmarks reflect standard practice and can be reproduced exactly; MBL, which has no maintained Python package, was assembled from scikit-learn components. Early stopping and hyperparameter selection used only training-fold data (Section 2.6), and every number reported in this paper is regenerated from stored per-fit records by a single script (Section 2.10).

**2.6 Tuning, training and random seeds**

Hyperparameters were selected inside each outer training fold (nested selection), using a group-disjoint inner split, so the outer test fold was never used for any decision. The inner unit was the same as the outer unit for the cluster and site designs, and the sentinel site for the country and agro-ecological designs, in which only three or four outer groups remain in training. Search spaces were sized to each model's hyperparameters (Table S2): one-dimensional grids for PLSR (7 values) and ridge (11), two-dimensional grids for SVR (9), KRR (9), Cubist (4) and LightGBM (2 leaf settings, with the number of trees set by early stopping), MBL (3 neighbourhood sizes), and three architectures for each neural model. For cheap models the inner score was the mean over three group-disjoint inner folds; for expensive models a single inner split holding out about 20% of the training samples by group was used. Neural networks were trained with Adam (Kingma and Ba, 2015), learning rates of 10⁻³ to 3 × 10⁻³ and weight decay, with early stopping on the same group-disjoint inner split (patience 10 epochs, maximum 80). Each neural model was trained from **five random initialisations** in every fold. Neural results are reported as the mean over seeds, with the seed standard deviation, and every bootstrap draw averages over seeds, so seed variability is carried into the confidence intervals.

**2.7 Validation designs**

Five designs were compared, ordered by how far the test samples lie from the training data (Table S1 lists the composition of each fold):

> \- **Random** (5 folds): samples assigned at random, ignoring grouping. This is included only to quantify the optimism of random splitting.
>
> \- **Leave-cluster-out** (5 folds): the 207 LDSF clusters assigned to folds, so no cluster contributes to both training and test. Because topsoil and subsoil samples from one profile belong to the same cluster, the grouping also keeps profile pairs together.
>
> \- **Leave-site-out** (8 folds): the 16 sentinel sites assigned to eight folds of one to three sites each, balanced by sample count.
>
> \- **Leave-AEZ-out** (5 folds): each Köppen–Geiger group held out in turn.
>
> \- **Leave-country-out** (4 folds): each country held out in turn.

In every design each fold serves once as the test fold, so every sample, and every country, is predicted from models that never saw it.

**2.8 Evaluation metrics and statistical tests**

Within each design every sample is predicted exactly once, so metrics were computed on the pooled out-of-fold predictions. This avoids the instability of R² computed on small folds with little variance. We report R², root mean square error (RMSE), RMSE divided by the training-set standard deviation, the ratio of performance to deviation (RPD = SD/RMSE), the ratio of performance to interquartile distance (RPIQ; Bellon-Maurel *et al.*, 2010), Lin's concordance correlation coefficient (CCC; Lin, 1989), and the mean absolute error skill relative to predicting the training-fold mean (MAE skill = 1 − MAE/MAE_mean). R² on the log scale is also given for skewed properties. Back-transformed predictions were limited to the range of the target in the training fold, identically for every model; Section 3.4 reports what happens without this safeguard. The range of per-fold R² is reported only as a description of dispersion. Confidence intervals come from a group bootstrap (2,000 draws) that resamples the outer spatial units (clusters for the random and cluster designs, sites otherwise) with replacement. Models were compared with paired group-bootstrap tests on the difference in pooled R², using the same draws for both models, and across properties with the Friedman test and the Nemenyi critical difference (Demšar, 2006). No metric is selected post hoc as a maximum over models; where a best model is named, it is the model with the best mean rank, and its paired comparisons with the other models are reported.

**2.9 Additional experiments**

*Ablation.* Under leave-site-out, each component of the hybrid was replaced in turn while everything else, including the selected configuration, was held fixed: (a) linear head; (b) dense ReLU head; (c) a fixed-centre Gaussian-activation head; (d) mean pooling instead of attention; (e) attention used to re-weight positions, followed by flattening, instead of attention pooling; (f) no positional embedding; (g) concatenation instead of the gate; (h) spectra only, without the ancillary branch; and (i) coordinates added to the ancillary input (also tested for LightGBM). Ablations used three seeds, except the linear head, which used five.

*Boundedness.* The hybrid was compared with the otherwise identical network with a linear head under the site, agro-ecological and country designs, on both the original and the log scale. The comparison used worst-case and lower-decile per-property R², and ‖**w**~o~‖₁ was reported.

*Calibration breadth.* (i) Like-for-like: each Kenyan site was predicted from the other two Kenyan sites (K2), from those sites plus all non-Kenyan samples (K2 + EA), from non-Kenyan samples only (EA), and from K2 + EA subsampled to the size of K2 (K2 + EA, size-matched; five draws). The test population is identical in every condition, and the size-matched condition separates diversity from sample size. Inner tuning splits were by site, except in the two-site Kenyan condition, where clusters were used. (ii) Learning curves: within each leave-site-out fold, training sets of n = 100, 200, 400 and 800 samples were drawn from m = 2, 3, 4, 6, 8, 10 or all training sites (three draws per cell), with the test fold fixed.

*Local spiking.* For each leave-site-out fold, k = 0, 5, 10, 20 or 50 samples from the withheld sites were added to the training set. Spike sets were nested, and every k was scored on the same test samples: the withheld samples, excluding the 50 candidates and any sample from the same profile. Three draws were made for PLSR, KRR and LightGBM and one for the hybrid.

*Conformal prediction.* Split-conformal intervals (Angelopoulos and Bates, 2023) at a nominal 90% level were calibrated on a set held out of each training fold and not used for fitting. Two calibration sets were compared: whole sites (group-aware) and a random 20% of training samples. Residuals were computed on the fitting scale and the intervals back-transformed.

*Interpretability.* For each leave-site-out hybrid model, the attention profile (spread over each position's receptive field) was compared with occlusion importance (an eight-bin window set to the training mean), integrated gradients (Sundararajan *et al.*, 2017) and PLSR variable importance in projection (VIP). Agreement was measured with Spearman correlation. The share of importance falling inside literature absorbance windows (Janik *et al.*, 1998; Nguyen *et al.*, 1991; Parikh *et al.*, 2014) was compared with 2,000 random circular shifts of the same profile, which preserve its autocorrelation. Modality contribution was estimated by permuting the ancillary input in the test fold and by the spectra-only ablation, not from the gate.

*Agronomic screening.* For properties with accepted interpretive classes for the extraction used (pH, organic C, Olsen P, exchangeable Ca, Mg and K, eCEC), out-of-fold leave-site-out predictions were used to classify samples against indicative critical levels (Landon, 1991). Sensitivity, specificity and balanced accuracy were computed. No deficiency diagnosis was attempted for quasi-total elements or total N, which do not measure plant-available fractions (Lindsay and Norvell, 1978).

**2.10 Implementation and reproducibility**

All code is in Python 3 (NumPy, pandas, scikit-learn [[sk_ver]], LightGBM [[lgb_ver]], PyTorch, cubist). Fold assignments are deterministic (seed 20260804); neural seeds are 11, 22, 33, 44 and 55. In total [[n_fits_total:,]] model fits were run. Each fit writes a record with the selected configuration, test indices and predictions, and every table, figure and number in this manuscript is regenerated from those records by a single script.
