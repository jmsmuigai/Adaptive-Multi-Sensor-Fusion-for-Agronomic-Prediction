# Adaptive Multi-Modal Fusion for Agronomic Prediction: An Attention-Weighted Hybrid CNN–RBF Model for Soil Nutrient Prediction Across East African Agro-Ecological Zones

**Running title:** Attention-weighted hybrid model for soil prediction

James Mukoma Mburu¹*, Kennedy Ogada¹, Michael Kimwele¹

¹ Department of Computing, School of Computing and Information Technology, Jomo Kenyatta University of Agriculture and Technology, P.O. Box 62000-00200, Nairobi, Kenya

*Corresponding author: jmsmuigai@gmail.com

---

## Abstract

Soil nutrient depletion is a primary constraint on smallholder productivity across Kenya and the wider East African region, and conventional wet-chemistry analysis is too costly to support the sampling densities that site-specific fertiliser advice requires. This study designs and rigorously evaluates an attention-weighted hybrid architecture for predicting soil nutrient status from mid-infrared (MIR) diffuse reflectance spectra fused with ancillary site information. The proposed model couples a one-dimensional convolutional feature extractor over the wavenumber axis, a Dynamic Attention Fusion Module (DAFM) that learns a per-sample weighting of spectral regions, a learned branch gate that balances the spectral and ancillary modalities, and a differentiable radial basis function (RBF) regression head, trained end to end. The architecture was evaluated on 1,310 georeferenced soil samples with paired MIR spectra and laboratory chemistry drawn from 16 Africa Soil Information Service sentinel sites in Kenya, Ethiopia, Tanzania and Uganda, against five benchmarks — partial least squares regression, ridge regression, random forest, gradient boosting and a multilayer perceptron — under three nested, group-disjoint validation designs. All models received an identical tuning budget and all preprocessing was fitted within training folds. Thirteen soil properties were predicted with mean R² up to 0.749 within landscapes, including total nitrogen at 0.939, pH at 0.916 and effective cation exchange capacity at 0.906. Predictive skill declined systematically as the withheld unit widened, from 0.749 under leave-cluster-out to 0.351 under leave-site-out and 0.205 under leave-country-out. The hybrid architecture achieved the highest mean R² of all six models when Kenya was withheld entirely (0.240), and its branch gate quantified the spectral contribution as ranging from 35.0% for exchangeable calcium to 59.2% for total nitrogen. Calibration breadth proved a stronger determinant of transferability than model architecture.

**Keywords:** precision agriculture; soil nutrient prediction; mid-infrared spectroscopy; convolutional neural networks; Dynamic Attention Fusion Module; radial basis function networks; agro-ecological zones; Kenya

---

## 1.0 Introduction

Agriculture underpins the food security and economic stability of a majority of households in sub-Saharan Africa. Despite this central role, smallholder farming systems consistently deliver yields well below their biophysical potential, and soil nutrient depletion is among the most persistent causes. Fertiliser recommendations across much of East Africa are still issued as blanket rates for large administrative units, even though soil fertility varies substantially over distances of a few hundred metres. Closing this gap requires soil information at a spatial density that conventional analysis cannot supply affordably: a full nutrient determination costs more than many smallholders would spend on fertiliser in an entire season.

Mid-infrared (MIR) diffuse reflectance spectroscopy offers the most credible route past this constraint. A soil spectrum encodes the vibrational absorbance of mineral and organic constituents, and a single measurement lasting under a minute can support simultaneous inference of several properties. The technique has been developed for tropical soils over two decades (Shepherd and Walsh, 2002), and the Africa Soil Information Service (AfSIS) has since assembled spectral libraries at continental scale (Vågen et al., 2016), providing the empirical foundation on which operational systems could be built.

Translating a spectral library into reliable agronomic advice is a modelling problem, and progress has followed the wider trajectory of machine learning. Partial least squares regression (PLSR) remains the operational default, handling the extreme collinearity of spectral data by projection onto a small number of latent components. Convolutional neural networks were subsequently shown to improve on PLSR in several settings, on the argument that convolution along the wavenumber axis captures the multi-band absorbance features characteristic of functional groups (Padarian et al., 2019; Ng et al., 2019). More recently, attention-based designs have come to dominate the literature: Guo et al. (2025) transform MIR spectra into two-dimensional representations for a Swin Transformer, Shuai et al. (2025) couple channel attention with a ConvNeXt backbone for visible–near-infrared data, and Peng (2026) applies self-attention to weight an ensemble of spectral learners dynamically.

Despite this progress, three methodological gaps persist at the intersection of classical and modern approaches, and they motivate the architecture developed here.

The first is **the treatment of kernel methods within differentiable pipelines**. Radial basis function models and support vector regressors remain well regarded in soil science for their resistance to overfitting on modest sample sizes, but conventional implementations rely on quadratic programming and yield decision functions that are not differentiable with respect to an upstream feature extractor. They therefore cannot be trained jointly with a neural encoder, and are typically bolted on as a separate stage that cannot adapt its representation.

The second is **the interpretability of attention**. Attention weights are frequently presented as making a network's reasoning transparent. For spectral data this claim is more defensible than for text or images, because a weight on a wavenumber has a direct physical referent. However, where attention operates over latent channels rather than input positions, no valid mapping back to the input exists, and relabelling latent dimensions with variable names is not justified. Attention weights are also rarely reported with uncertainty or validated against an independent measure.

The third is **validation design**. Soil spectral libraries are assembled from clustered sampling campaigns in which a survey team visits a site and collects many samples within it. Random splitting places near-duplicate samples on both sides of the partition, so the resulting accuracy describes interpolation among neighbours rather than prediction at a new location. The consequences of ignoring grouped structure are well documented in ecological and geographic modelling (Roberts et al., 2017; Ploton et al., 2020), and a dedicated literature addresses the difficulty of moving calibrations between instruments and regions (Gamagedara et al., 2025). Studies reporting architectural gains are nonetheless typically evaluated within a single region, leaving open whether the reported ranking survives extrapolation.

This study addresses all three. We design an attention-weighted hybrid architecture in which a differentiable RBF head permits end-to-end training, attention operates over spectral positions so that weights remain attributable to absorbance regions, and a learned branch gate reports rather than assumes the contribution of each input modality. We then evaluate it against five benchmarks under three nested, group-disjoint validation designs, with equal tuning for every model. Specifically we ask:

1. Can a differentiable RBF head, trained jointly with a convolutional extractor and an attention fusion module, predict soil nutrient status competitively with established methods?
2. How does the predictive skill of each model change as the withheld spatial unit widens from a sampling cluster to a sentinel site to an entire country?
3. Do the learned attention weights localise onto chemically interpretable absorbance regions, and can the contribution of each modality be measured rather than assumed?
4. Which soil properties are simultaneously limiting for crop production in these landscapes and reliably predictable from spectra?

We report each outcome as the evidence supports it, including where the results do not favour the proposed architecture.

## 2.0 Materials and Methods

### 2.1 Study area

Samples were drawn from 16 AfSIS Land Degradation Surveillance Framework (LDSF) sentinel sites distributed across four East African countries. Each site is a 10 × 10 km block sampled on a randomised, spatially stratified hierarchical design in which clusters are placed within the block and plots within clusters (Vågen et al., 2010). Sites were allocated by stratification on Köppen–Geiger climate zones, so the collection spans a wide environmental range rather than constituting a convenience sample.

The four countries contribute as summarised in Table 1 and mapped in Figure 1. Kenya contributes three sites spanning a pronounced agro-ecological gradient: Morijo in the highland Rift Valley of Narok County, Mbalambala in the arid eastern lowlands of Garissa County, and Marafa in the coastal hinterland of Kilifi County. Ethiopia contributes four sites in the northern highlands, Tanzania seven across the central and southern plateau, and Uganda two in the Lake Victoria basin. Together the samples span approximately 22 degrees of latitude and 12 of longitude, and encompass humid highland, sub-humid, semi-arid, arid and coastal settings. Field sampling took place between 2009 and 2013 at two depths, topsoil (0–20 cm, n = 659) and subsoil (20–50 cm, n = 651).

[[TABLE 1]]

[[FIGURE 1]]

The 1,310 samples are a stratified sample of landscapes rather than a probability sample of East African cropland. This distinction matters for how the agronomic prevalence figures in Section 3.6 should be read, but it is precisely what makes the nested transfer analysis in Section 3.2 interpretable.

### 2.2 Data sources and dataset assembly

Two openly licensed AfSIS deposits were joined on the soil sample number (SSN).

**Spectra.** Vågen et al. (2020) provide MIR spectra for approximately 18,500 AfSIS Phase I samples, measured on a Bruker Tensor 27/HTs-XT FT-IR at the ICRAF Soil-Plant Spectral Diagnostics Laboratory in Nairobi and distributed as one file per country. The Kenya, Ethiopia, Tanzania and Uganda files together hold 4,654 spectra across 1,749 wavenumbers from 601.7 to 4001.6 cm⁻¹ at 1.9 cm⁻¹ spacing.

**Reference chemistry.** Vågen et al. (2021) provide wet-chemistry determinations for 2,002 archived AfSIS Phase I samples, analysed at Rothamsted Research in 2017–2018, of which 1,310 belong to the four study countries.

The assembly procedure verifies that every country file shares an identical wavenumber grid and halts if it does not, so spectra acquired under differing configurations are never silently concatenated. **The join is complete: all 1,310 chemistry records matched a spectrum, with zero attrition.** This is stated explicitly because incomplete joins are a common and rarely disclosed source of selection bias.

Thirteen properties were modelled (Table 2): pH in water at 1:2.5; organic carbon and total nitrogen by dry combustion; Olsen-extractable phosphorus; exchangeable Ca, Mg, K and effective cation exchange capacity by cobalt hexamine extraction with ICP-OES; and quasi-total S, Fe, Mn, Zn and Cu by aqua regia digest with ICP-OES. Missing values are negligible, ranging from none to four per property.

[[TABLE 2]]

One defect in the source metadata is recorded for transparency: the deposit's variable dictionary transposes the descriptions of `Percent_N` and `Percent_C`. The column names are correct, as the observed medians confirm, and the names rather than the descriptions were used.

### 2.3 Data preprocessing and spectral harmonisation

Spectral preprocessing follows established chemometric practice. Critically for the validity of the reported metrics, **every data-dependent step is fitted on training-fold rows only** and applied unchanged to the held-out fold:

1. **Savitzky–Golay smoothing with a first derivative** (window 11 points, second-order polynomial), suppressing additive baseline drift and sharpening overlapping absorbance features.
2. **Standard Normal Variate (SNV) transformation**, removing the multiplicative scatter introduced by variation in particle size and packing density.
3. **Binning by a factor of four**, reducing 1,749 bands to 437 while preserving spectral shape and lowering the parameter count of the convolutional models.
4. **Per-band standardisation** to zero mean and unit variance using training-fold statistics.

Target variables with skew exceeding 2.0 were log₁ₚ-transformed for fitting; all predictions were back-transformed before any metric was computed, so every RMSE and MAE reported here is expressed in the original measurement unit of the property concerned.

### 2.4 Proposed architecture: Hybrid Attention-Weighted CNN–DAFM–RBF

The proposed model comprises two input branches, an attention fusion module and a differentiable kernel-style regression head, integrated into a single end-to-end differentiable pipeline (Figure 2).

[[FIGURE 2]]

#### 2.4.1 Spectral branch: 1D convolutional feature extractor

The spectral branch applies two one-dimensional convolutional layers with ReLU activation and intervening max-pooling along the wavenumber axis, producing a feature map **H** ∈ ℝ^(p×C) where p indexes retained spectral positions and C the learned channels. Convolution is an appropriate inductive bias in this setting because wavenumbers form an ordered sequence in which neighbouring bands are physically related through the width of absorbance features. The convolutional layer output at position *j* and channel *l* is

&nbsp;&nbsp;&nbsp;&nbsp;*h*ⱼ,ₗ = σ ( Σₖ *w*ₖ,ₗ *x*ⱼ₊ₖ₋₁ + *b*ₗ )  (Eq. 1)

where σ denotes the ReLU nonlinearity, *w* the kernel weights and *k* the kernel index.

#### 2.4.2 Dynamic Attention Fusion Module

The DAFM computes additive attention over the p spectral positions, producing a sample-specific weighting of absorbance regions:

&nbsp;&nbsp;&nbsp;&nbsp;*e*ⱼ = **v**ᵀ tanh(**W** **h**ⱼ + **b**)  (Eq. 2)

&nbsp;&nbsp;&nbsp;&nbsp;α ⱼ = exp(*e*ⱼ) ⁄ Σₖ exp(*e*ₖ)  (Eq. 3)

&nbsp;&nbsp;&nbsp;&nbsp;**z**ₛ = Σⱼ α ⱼ **h**ⱼ  (Eq. 4)

where **h**ⱼ ∈ ℝ^C is the channel vector at position *j*, **W** ∈ ℝ^(C×A) and **v** ∈ ℝ^A are learned, α sums to one over positions, and **z**ₛ ∈ ℝ^C is the resulting spectral context vector.

The decision to attend over **spectral positions rather than latent channels** is deliberate and consequential for interpretation. Each position corresponds to a contiguous wavenumber window, so α ⱼ is attributable to an identifiable absorbance region and can be plotted directly against the spectrum (Section 3.5). Attention computed over latent channels admits no such mapping and cannot legitimately be relabelled with input variable names.

#### 2.4.3 Ancillary branch and learned modality gate

The ancillary branch is a dense encoder over sampling depth and geographic position, producing **z**_c. A learned gate then weighs the two modalities against one another for each sample:

&nbsp;&nbsp;&nbsp;&nbsp;**g** = softmax(**W**_g [**z**ₛ ; **z**_c] + **b**_g)  (Eq. 5)

&nbsp;&nbsp;&nbsp;&nbsp;**z** = [*g*₁**z**ₛ ; *g*₂**z**_c]  (Eq. 6)

Because **g** is an output of the fitted model, the relative contribution of each modality is a measured quantity that can be reported with fold-level uncertainty rather than asserted. This provides a direct, quantitative account of what multi-modal fusion delivers for each property.

#### 2.4.4 Differentiable RBF regression head

Conventional radial basis function regressors and support vector regressors rely on quadratic programming solvers and yield decision functions that are not differentiable with respect to an upstream encoder. To resolve this, the head is implemented as a dense projection terminating in a radial basis activation with a learnable per-unit precision γ:

&nbsp;&nbsp;&nbsp;&nbsp;**p** = **W**_r **z** + **b**_r  (Eq. 7)

&nbsp;&nbsp;&nbsp;&nbsp;φᵢ = exp(−γᵢ *p*ᵢ²)  (Eq. 8)

&nbsp;&nbsp;&nbsp;&nbsp;ŷ = **W**_o **φ** + *b*_o  (Eq. 9)

Because every operation from the input tensors through the RBF activation to the linear output is differentiable with respect to its parameters, the complete CNN–DAFM–RBF superstructure trains end to end by gradient descent, and the convolutional representation adapts to the requirements of the kernel-style head rather than being fixed in advance.

We state explicitly that this head is **not** a support vector regressor. It has no ε-insensitive loss, no margin, no dual coefficients and no support vectors. It is a radial-basis-activated neural regression head, and is named as such throughout.

#### 2.4.5 Mathematical properties of the architecture

Three properties of the design follow directly from its construction and are stated here because they predict the behaviour observed empirically in Section 3.2.

**Proposition 1 (Attention performs convex pooling).** The softmax in Eq. 3 satisfies α ⱼ > 0 and Σⱼ α ⱼ = 1, so α lies in the probability simplex Δ^(p−1). The context vector is therefore a convex combination of the position-wise channel vectors,

&nbsp;&nbsp;&nbsp;&nbsp;**z**ₛ ∈ conv{**h**₁, …, **h**_p},  hence  ‖**z**ₛ‖∞ ≤ maxⱼ ‖**h**ⱼ‖∞  (Eq. 10)

Attention pooling can therefore never produce a context vector outside the range of the convolutional features it aggregates, whatever the input. This distinguishes it from an unconstrained linear projection of the same features, whose norm is unbounded in the feature magnitude.

**Proposition 2 (The prediction is bounded).** Because γᵢ > 0 by construction (γᵢ = exp(logγᵢ)), each RBF unit satisfies 0 < φᵢ = exp(−γᵢ *p*ᵢ²) ≤ 1, with equality only at *p*ᵢ = 0. The network output in Eq. 9 is therefore bounded for every input:

&nbsp;&nbsp;&nbsp;&nbsp;| ŷ − *b*_o | = | **W**_o **φ** | ≤ ‖**W**_o‖₁ ‖**φ**‖∞ ≤ ‖**W**_o‖₁  (Eq. 11)

so ŷ ∈ [ *b*_o − ‖**W**_o‖₁ , *b*_o + ‖**W**_o‖₁ ] regardless of how far the input lies from the calibration domain. By contrast, a linear predictor ŷ = **w**ᵀ**x** + *b* satisfies |ŷ| → ∞ as ‖**x**‖ → ∞, and an unbounded neural head with ReLU activations and a linear output layer has the same property. This is a structural, not an empirical, difference.

**Proposition 3 (Gradient saturation away from the calibration domain).** Differentiating Eq. 8,

&nbsp;&nbsp;&nbsp;&nbsp;∂φᵢ ⁄ ∂*p*ᵢ = −2 γᵢ *p*ᵢ exp(−γᵢ *p*ᵢ²)  (Eq. 12)

whose magnitude attains its maximum √(2γᵢ ⁄ e) at *p*ᵢ = ±(2γᵢ)^(−1/2) and decays to zero as |*p*ᵢ| → ∞. A projection lying far from the region the unit was fitted to therefore contributes vanishing gradient, and the unit's output saturates toward zero rather than growing. The head effectively abstains on inputs unlike its training data instead of extrapolating through them.

Taken together, Propositions 1–3 imply that the proposed architecture cannot fail without bound. Its worst-case error is controlled by ‖**W**_o‖₁, a quantity fixed at training time, whereas linear and unbounded neural predictors have no such control. Section 3.2 reports the empirical consequence: under leave-site-out validation the worst single-property R² was −24.15 for the multilayer perceptron, −2.10 for PLSR and −1.64 for ridge regression, against −0.43 for the proposed architecture. Only gradient boosting, whose predictions are convex combinations of training-leaf values and are therefore also bounded, achieved a better worst case (−0.04). The models that degrade gracefully are precisely those whose output range is bounded by construction, which is what Proposition 2 predicts.

### 2.5 Training procedure

Optimisation used Adam (Kingma and Ba, 2015) with early stopping on a held-out portion of the training fold. The learning rate, channel widths, kernel sizes, attention dimension and number of RBF units were selected from the tuning grid described in Section 2.6. Analytic gradients for every parameter in the network were verified against central finite differences to a worst-case relative error of 1.45 × 10⁻⁶, confirming the correctness of the backward pass prior to any experiment.

### 2.6 Benchmark models and tuning protocol

Five benchmarks were implemented for comparison: PLSR by the NIPALS algorithm, the chemometric reference method; ridge regression on the preprocessed spectra; a histogram-based random forest; histogram-based gradient boosting; and a multilayer perceptron.

**Every model, including the proposed architecture, received an identical tuning budget of three candidate configurations**, scored on a reserved fold, with the selected configuration then applied unchanged across evaluation folds. Equal budgets matter for the credibility of any architectural comparison: a substantial part of the advantage reported for novel architectures in the literature can arise from unequal tuning effort rather than from the architecture itself.

### 2.7 Validation designs

The central experiment holds the data, features, preprocessing and tuning budget fixed and varies only the spatial scale of the withheld group. All three designs are group-disjoint.

- **Leave-cluster-out** (5 folds). The 207 LDSF clusters are assigned to folds so that no cluster contributes samples to both training and test. This corresponds to predicting at a new location within an already-surveyed landscape.
- **Leave-site-out** (8 folds). The 16 sentinel sites are assigned whole to folds, so the model must predict within a 10 × 10 km block from which it has seen nothing. This corresponds to extending an existing library to a new landscape.
- **Leave-country-out** (4 folds). Each national dataset forms one fold. This is the most demanding design and corresponds to exporting a calibration across a national border.

In every design, fold 0 is reserved exclusively for hyperparameter selection and is never used as a test fold, so no reported metric is contaminated by model selection.

### 2.8 Evaluation metrics

For each property, design and model we report R², RMSE, MAE, bias and normalised RMSE, each as a mean over evaluation folds with the standard deviation and a t-based 95% confidence interval. We additionally report, for each design, the mean rank of each model across the thirteen properties and the number of properties for which it retains R² > 0, because mean R² can be dominated by a single catastrophic fold and is therefore an incomplete summary on its own.

### 2.9 Threats to validity and the controls applied

Because the central claims of this study concern generalisation, we set out explicitly the ways in which a spectral modelling study of this kind can produce optimistic results, and the control applied to each.

**Grouped-sample leakage.** Samples collected within one cluster or site are near-duplicates. Control: all three validation designs are group-disjoint at a stated spatial scale, and no random split is reported anywhere in this paper.

**Preprocessing leakage.** Fitting a scaler, imputer or transform on the full dataset lets test-fold information influence the training representation. Control: every data-dependent preprocessing step — median statistics, band standardisation, target scaling and the log transform decision — is fitted on training-fold rows alone.

**Model-selection leakage.** Selecting hyperparameters on a fold that is later used as a test fold inflates the reported metric. Control: fold 0 is reserved for selection in every design and never contributes to a reported number.

**Unequal tuning.** A novel architecture given more search effort than its baselines will appear superior for reasons unrelated to its design. Control: all six models receive exactly three candidate configurations, scored identically.

**Metric fragility.** A single catastrophic fold can dominate a mean R², and a single favourable one can flatter a model. Control: fold standard deviations and 95% confidence intervals accompany every estimate, and rank-based and count-based summaries are reported alongside means.

**Implementation error.** A subtly incorrect gradient can train a model that appears to work while optimising the wrong objective. Control: analytic gradients for every parameter of the proposed network were checked against central finite differences before any experiment was run.

**Selective reporting.** Reporting only the configuration or subset that favours a conclusion is difficult for a reader to detect. Control: results for all six models, thirteen properties and three designs are reported in full, and an automated script re-derives every numeric claim in this manuscript from the stored per-fold records.

We do not claim these controls are exhaustive. They address the failure modes we could identify, and the released artefacts allow any reader to test for others.

### 2.10 Implementation and reproducibility

All models, cross-validation designs, preprocessing routines and metrics were implemented directly in NumPy without machine-learning framework dependencies, so the complete analysis reproduces from a bare Python installation. A single random seed (20260804) governs fold assignment, bootstrap sampling, weight initialisation and minibatch shuffling. In total 1,794 model fits were executed, each writing an individual record containing the selected configuration, test indices, predictions and metrics. An accompanying Google Colab notebook reproduces the full pipeline from data download to figures.

## 3.0 Results and Discussion

### 3.1 Within-landscape prediction performance

Table 3 and Figure 3 present leave-cluster-out results. Prediction is strong for the majority of properties. The best model achieved R² = 0.939 for total nitrogen (95% CI 0.924–0.954), 0.916 for pH, 0.915 for total iron, 0.906 for effective cation exchange capacity, 0.900 for exchangeable calcium and 0.897 for organic carbon. Exchangeable magnesium (0.811) and total zinc (0.810) follow, then total copper (0.713), total manganese (0.626), Olsen phosphorus (0.562), total sulphur (0.513) and exchangeable potassium (0.504).

[[TABLE 3]]

[[FIGURE 3]]

Averaged across the thirteen properties, mean R² was **MLP 0.749, gradient boosting 0.732, random forest 0.695, ridge 0.685, the proposed hybrid 0.636, and PLSR 0.505**.

Three observations follow. First, **every nonlinear method outperformed PLSR**, by between 0.13 and 0.24 mean R². Given that PLSR remains the operational default in soil spectroscopy laboratories, this is a practically consequential finding: on a library of this size, any of the four alternatives would yield materially better calibrations at negligible additional computational cost. PLSR fell below zero for total sulphur (−0.848), where its linear projection is evidently unable to accommodate the extreme skew of that property.

Second, **the proposed hybrid architecture ranked fifth of six within landscapes**, with a mean rank of 4.92. We had anticipated that convolutional treatment of the ordered spectral sequence, combined with attention pooling and a kernel-style head, would confer an advantage in this regime, and it did not. Its absolute performance is nonetheless substantial — it exceeded R² = 0.80 for five properties and 0.70 for seven — but a plain multilayer perceptron on the same preprocessed spectra performed better for ten of the thirteen properties at a considerably smaller parameter count. With 1,310 samples, the additional capacity of the convolutional and attention components appears not to be recoverable from the data.

Third, and of direct agronomic consequence, **Olsen phosphorus reached R² = 0.562**. In a preliminary analysis restricted to the three Kenyan sites, the best achievable value for this property was 0.164 with a confidence interval spanning zero. This improvement is not attributable to any change in model or preprocessing; it follows from the fourfold increase in calibration samples and the fivefold increase in landscapes represented, and is developed in Section 3.3.

Figure 4 shows out-of-fold predictions against observations for the four best-predicted properties, pooled across evaluation folds.

[[FIGURE 4]]

### 3.2 Model behaviour across nested validation designs

Table 4 and Figure 5 present the three designs together. Best-model mean R² across the thirteen properties falls from **0.749 under leave-cluster-out, to 0.351 under leave-site-out, to 0.205 under leave-country-out**.

[[TABLE 4]]

[[FIGURE 5]]

This ordering is not an artefact of the analysis. Each successive design withholds a larger and more environmentally distinct block, so the model is required to extrapolate progressively further from its calibration domain. The magnitude of the decline is the practically useful quantity: approximately half the apparent skill of a within-landscape calibration is lost when the model is applied to a new landscape, and about 40% of what remains is lost again when it crosses a national border. Any single reported accuracy for a soil spectral calibration is therefore uninterpretable without an accompanying statement of the validation design that produced it.

Property-level behaviour under leave-site-out (Table 5) is informative. Organic carbon (R² = 0.814), total nitrogen (0.670), effective CEC (0.665), total iron (0.639) and exchangeable calcium (0.550) transfer well, reflecting strong and spatially consistent MIR expression. Exchangeable potassium (0.137) and Olsen phosphorus (0.032) barely transfer at all.

[[TABLE 5]]

**Gradient boosting proved the most reliable model across designs**, achieving the best mean rank under both leave-site-out (2.31) and leave-country-out (2.69). The multilayer perceptron, strongest within landscapes, was brittle under transfer: its leave-site-out mean R² of −1.791 is driven by a single catastrophic fold for total iron (−24.1), which illustrates why mean R² should never be reported without fold-level dispersion or a rank-based summary alongside it.

The pattern of worst-case failures is not arbitrary, and Table 6 sets it against the bound derived in Section 2.4.5. The three models whose output is unbounded in the input — the multilayer perceptron with its linear output layer, PLSR and ridge regression — produced the three most severe failures (−24.15, −2.10 and −1.64). The three whose output is bounded by construction — gradient boosting and random forest, whose predictions are convex combinations of training-leaf values, and the proposed architecture, whose RBF head satisfies |ŷ − *b*_o| ≤ ‖**W**_o‖₁ by Proposition 2 — produced the three least severe (−0.04, −0.47 and −0.43). This is what Propositions 2 and 3 predict: a bounded, saturating head cannot diverge on inputs unlike its training data, it can only revert toward its bias term. The proposed architecture is the only *neural* model in the bounded group, and its worst case is 57 times smaller than that of the multilayer perceptron trained on identical features.

[[TABLE 6]]

The proposed hybrid architecture sat mid-pack under both transfer designs, ranking third of six under leave-site-out, while retaining positive skill for 11 of 13 properties — comparable to the tree ensembles and better than PLSR, ridge or the MLP.

Figure 6 shows fold-level dispersion for every model and property. For well-predicted properties the four folds agree closely; for Olsen phosphorus, sulphur and potassium the fold-to-fold range frequently exceeds the difference between models, so reporting a single pooled R² for these would convey more precision than the data support.

[[FIGURE 6]]

### 3.3 Effect of calibration breadth on transferability

The comparison in Table 7 is, in our assessment, the most consequential result of the study for operational practice. Holding models, preprocessing, tuning budget and validation design constant, and varying only the geographic base of the calibration, restricting the analysis to the three Kenyan sites produced **negative mean R² for every model** under leave-site-out validation: none could usefully predict within a withheld landscape. The four-country calibration achieved a positive mean of 0.351 and retained useful skill for twelve of thirteen properties.

[[TABLE 7]]

The magnitude of this effect should be set against the differences between models. Within the four-country analysis, the gap between the best and second-best model under leave-site-out is 0.047 R², and between gradient boosting and the best linear method 0.377. The change attributable to calibration breadth is approximately 1.0. Expanding the geographic base of a spectral library therefore delivers substantially more benefit than switching between reasonable model families.

This finding also reframes the recent architectural literature. Studies reporting improvements from transformer or attention-based designs on single-region libraries (Guo et al., 2025; Shuai et al., 2025; Peng, 2026) may be optimising a factor of secondary importance relative to the composition of the calibration set itself.

### 3.4 Transfer performance resolved by country

Aggregate leave-country-out figures conceal substantial variation between national datasets. Table 7 resolves performance by held-out country and Figure 7 presents it graphically. Ethiopia served as the hyperparameter-selection fold and is therefore not reported as a test country.

[[TABLE 8]]

[[FIGURE 7]]

Best-model mean R² was **0.350 for Tanzania, 0.240 for Kenya and 0.193 for Uganda**, with 10, 9 and 10 of 13 properties respectively retaining positive skill. Tanzania is the easiest country to predict into and Uganda the hardest, consistent with their representation in the calibration: Tanzania contributes seven sentinel sites and Uganda only two, so predicting into Uganda requires the model to work from the fewest analogous landscapes.

Notably, **the proposed hybrid architecture achieved the highest mean R² of all six models when Kenya was withheld entirely** (0.240, against 0.188 for ridge regression and 0.120 for gradient boosting). Since Kenya is the primary area of application for this work, that result is of direct relevance. We do not over-read it: with three test countries a single country-level ordering rests on far less evidence than the pooled comparison in Section 3.2, which places gradient boosting first. The variation nonetheless cautions against assuming that a model selected on one national dataset remains the best choice on another.

Property-level behaviour under leave-country-out (Table 9) mirrors the leave-site-out pattern at lower absolute values. Organic carbon, total nitrogen and effective CEC remain the most transferable; exchangeable potassium and Olsen phosphorus remain essentially untransferable. That the same properties succeed and fail at both spatial scales suggests the limitation is intrinsic to the spectral expression of those analytes rather than to any particular geographic contrast.

[[TABLE 9]]

**The leave-country-out design is unbalanced, and the reader should weight it accordingly.** The four national datasets differ substantially in the number of sentinel sites they contribute — Tanzania 7, Ethiopia 4, Kenya 3, Uganda 2 — so the folds are not equally difficult and are not exchangeable. Withholding Tanzania removes seven landscapes from the training set but leaves the model with nine; withholding Uganda removes only two but requires prediction into a region with the fewest analogues in training. The design therefore confounds two distinct quantities: how much information is lost by withholding a country, and how distinctive that country's soils are relative to the remainder.

The observed ordering is consistent with the first of these dominating. Tanzania, with the most sites, is both the most costly to withhold and the easiest to predict into; Uganda, with the fewest, is the cheapest to withhold and the hardest to predict into. We therefore read the leave-country-out results as indicative of the general magnitude of cross-border degradation rather than as a ranking of national difficulty, and we do not claim that Ugandan soils are intrinsically harder to model than Tanzanian ones. Disentangling the two effects would require either balanced site counts or a resampling design in which the number of training sites is held constant while their national composition varies.

Two remedies are available and we state them explicitly for anyone extending this work. The first is to add sentinel sites from the under-represented countries: the AfSIS deposit contains further national spectral files, and the wet-chemistry deposit covers eighteen countries in total, so a more balanced panel can be assembled from the same open sources without new field campaigns. The second is to hold the number of training sites fixed — for instance, training on a random eight sites drawn from three countries and testing on the fourth, repeated across draws — which separates the effect of calibration size from the effect of national distinctiveness at the cost of a larger compute budget. We regard the first as the more informative of the two, because a larger and better-balanced panel improves the leave-site-out analysis simultaneously.

By contrast, the leave-site-out design in Section 3.2 does not suffer from this imbalance in the same way: its 16 folds are individual sentinel sites of comparable size, and it is on that design rather than the country design that we base the study's principal conclusions about transferability.

### 3.5 Residual behaviour and computational cost

Figure 8 presents residual diagnostics for the three best-predicted properties. Residuals plotted against fitted values show mild regression toward the mean at the extremes of each distribution, which is expected under squared-error training and matters for any application targeting unusually deficient or unusually rich soils. Residual distributions separated by country are approximately centred for Kenya, Ethiopia and Tanzania, with a modest offset for Uganda, the country contributing fewest sites. This pattern is consistent with the leave-country-out results in Section 3.4 and indicates that country-level bias should be checked before a library is applied in a new nation.

[[FIGURE 8]]

Computational cost was measured on the assembled dataset (1,310 samples, 437 preprocessed bands) on an aarch64 Linux virtual machine with 3.8 GiB RAM, Python 3.10.12 and NumPy 2.2.6, constrained to a single BLAS thread, with 30 timed repetitions following a warm-up pass (Table 10). Training the proposed architecture took 20.9 s and inference required 0.0785 ms per sample (p95 0.0875 ms), equivalent to approximately 12,700 samples per second. Gradient boosting trained in 2.6 s and inferred at 0.0354 ms per sample; the multilayer perceptron trained in 1.8 s and inferred at 0.0015 ms per sample; the linear methods were effectively instantaneous.

[[TABLE 10]]

All models are therefore fast enough for operational use at national scale on ordinary hardware: even the most expensive would process a year's throughput for a busy soil laboratory in seconds. We emphasise that these figures cover model inference only. They exclude sample preparation, spectrometer acquisition time and the laboratory workflow surrounding them, which dominate the real cost of any deployment, and no claim of operational readiness is made on the basis of them.

### 3.6 Interpretability of the attention module and modality gate

Figures 9 and 10 present the interpretability analysis. Mean attention over wavenumbers is non-uniform and concentrates in regions with recognised spectroscopic assignments. For total nitrogen the highest weights fall near 1445 cm⁻¹ and 2185 cm⁻¹; for organic carbon near 1861 cm⁻¹, 1584 cm⁻¹ and 2454 cm⁻¹, the latter two coinciding with carbonyl and aromatic C=C absorbance; for exchangeable calcium near 1476 cm⁻¹ and 1337 cm⁻¹, consistent with carbonate; and for pH the distribution is flatter and shifted toward 2639 cm⁻¹ and 3395 cm⁻¹ in the O–H stretching region.

[[FIGURE 9]]

The learned gate provides a quantitative account of modality contribution that depends on no post-hoc attribution method. Averaged across evaluation folds, the spectral branch received **59.2 ± 10.8% of the weight for total nitrogen, 55.5 ± 14.9% for organic carbon, 41.5 ± 8.9% for pH and 35.0 ± 16.1% for exchangeable calcium** (Table 11). That the model routes a majority of weight to spectra for the organically bound properties, and less for calcium, is chemically coherent: organic carbon and nitrogen have direct and strong MIR absorbance, whereas exchangeable calcium is more strongly conditioned by parent material and landscape position, information carried by the ancillary branch.

[[TABLE 11]]

[[FIGURE 10]]

We are deliberate about what this does and does not establish. The weights describe where the network routes information within its own representation. They are not causal statements, and they do not substitute for a controlled importance estimate obtained by perturbation. Their value lies in being read directly from the fitted model, reported with fold-level uncertainty, and attributable to physically meaningful axes — the wavenumber and the input modality — rather than to anonymous latent dimensions. This is the principal interpretability contribution of the architecture, and it holds regardless of where the model ranks on accuracy.

### 3.7 Agronomic status and the predictability gap

Applying indicative critical levels to the 659 topsoil samples (Table 12, Figure 11), the most widespread deficiencies are **Olsen phosphorus (84.4% of topsoils below the reference level), total copper (71.3%), total nitrogen (69.1%), total zinc (66.9%) and organic carbon (60.6%)**. Exchangeable bases are comparatively adequate: potassium falls below its critical level in 15.9% of topsoils, effective CEC in 14.6%, calcium in 8.5% and magnesium in 7.1%, and only 7.9% of samples are more acid than pH 5.5.

[[TABLE 12]]

[[FIGURE 11]]

Because published critical levels vary with source, extraction method and crop, Table 12 also reports the deficient fraction at half and one and a half times each threshold. The ranking of properties is stable across that range even though the absolute percentages are not, so the ordering is robust to the specific values adopted.

Setting this against predictability produces the study's most practically awkward finding. **Olsen phosphorus is the most widely deficient property and among the least transferable**: R² = 0.562 within landscapes, but only 0.032 under leave-site-out. Total nitrogen and organic carbon, also widely deficient, are both well predicted and reasonably transferable, which is encouraging for carbon and nitrogen monitoring. But phosphorus is where the fertiliser decision most often turns, and a spectral library cannot at present supply it in an unsampled landscape.

Two implications follow for deployment. Where phosphorus advice is the objective, spectroscopy should be paired with a modest number of wet-chemistry determinations in each new landscape rather than relied upon alone; the spectral model remains valuable for stratifying where those determinations should be taken. And aggregate accuracy across a panel of properties is a misleading summary of a library's agronomic usefulness: a calibration can appear excellent on average while failing on precisely the property that drives the recommendation.

### 3.8 Why phosphorus resists spectral prediction, and four routes to closing the gap

Because the phosphorus result is the principal practical limitation of this work, it warrants a mechanistic explanation and a concrete set of remedies rather than a statement of the deficit alone.

**The physical cause.** Mid-infrared spectroscopy measures molecular vibrations. Organic carbon and total nitrogen are predicted well because C–H, C=O, N–H and amide bonds absorb strongly and directly in the mid-infrared, so the analyte itself generates the signal. Phosphorus in soil is dominated by orthophosphate adsorbed onto iron and aluminium oxide surfaces and by occluded and organic pools. The P–O stretch near 1000–1100 cm⁻¹ lies beneath the intense and highly variable Si–O absorbance of the clay mineral matrix, which is one to two orders of magnitude stronger. What a calibration therefore learns is not a phosphorus signal but a correlation between phosphorus and the mineralogical and organic-matter context in which it sits.

That distinction explains the pattern in our results precisely. Within a landscape, where mineralogy and management are relatively homogeneous, the correlation holds and R² reaches 0.562. Transferred to a landscape with different parent material, the correlation does not hold, and R² collapses to 0.032. The model was never measuring phosphorus; it was measuring its neighbourhood.

**Route 1: extraction-specific modelling.** Our target is Olsen-extractable P, a bicarbonate extraction developed for calcareous and neutral soils. Many of the samples here are acidic, where Bray-1 or Mehlich-3 are the appropriate extractants and where Olsen systematically under-recovers. Part of the apparent unpredictability is therefore an artefact of applying one extraction convention across soils of differing pH and mineralogy: the target variable itself changes meaning across the transfer domain. Modelling extraction methods separately, or including pH and exchangeable acidity as explicit conditioning variables in the phosphorus model rather than pooling all soils into one calibration, would remove a source of variance that no amount of spectral modelling can absorb.

**Route 2: complementary sensing.** Because the limitation is physical, the productive response is to add a sensing modality in which phosphorus is directly detectable rather than to refine the model. Energy-dispersive X-ray fluorescence measures total elemental phosphorus directly and is now available in portable instruments at a cost comparable to a benchtop spectrometer. Total P is not the same quantity as available P, but XRF total P combined with MIR-derived pH, organic carbon and texture provides two largely independent constraints on the available fraction. The dual-branch architecture developed here is directly suited to that fusion: an XRF channel would enter as a second measured branch alongside the spectral one, and the learned modality gate would report how much each contributes for each soil, exactly as it currently does for the spectral and ancillary branches.

**Route 3: spike calibration with a small local sample.** The transfer literature shows that adding a modest number of locally analysed samples to a global calibration recovers a large part of the lost accuracy, and calibration-transfer methods formalise this (Gamagedara et al., 2025). For an operational laboratory the practical recipe is to analyse a small set of samples by wet chemistry in each new landscape — the spectral model itself can select which, by identifying samples that span the spectral variation of the new area — and add them to the training set. This converts an intractable zero-shot transfer problem into a tractable few-shot one, at a cost of a handful of determinations rather than hundreds.

**Route 4: stratified sampling guided by the spectral model.** Even where phosphorus cannot be predicted, the spectral model retains value for deciding *where* to spend a wet-chemistry budget. Because it predicts organic carbon, pH, texture and CEC accurately, it can partition a landscape into spectrally coherent strata, and phosphorus determinations can then be allocated across those strata rather than on an arbitrary grid. This does not predict phosphorus, but it makes each expensive determination more informative about the landscape as a whole.

We regard Routes 1 and 3 as immediately actionable with existing infrastructure, and Route 2 as the more substantial methodological opportunity — and the one that would most directly extend the multi-modal architecture presented here.

### 3.9 Comparison with published results

Placing these figures against the literature requires care, because published accuracies are seldom accompanied by a statement of validation design, and Section 3.2 shows that design alone can shift R² by more than 0.5.

Hengl et al. (2017) mapped soil nutrients across sub-Saharan Africa from remote-sensing covariates using random five-fold cross-validation, reporting R² between 0.40 and 0.85 for most nutrients and identifying phosphorus, sulphur and boron as the poorest performers. Our within-landscape results (0.50 to 0.94) are higher, as expected: MIR spectra measure the sample itself rather than inferring properties from environmental proxies. Our phosphorus result mirrors theirs, and the fact that two very different predictor sets fail on the same analyte strengthens the case that the limitation is intrinsic to phosphorus rather than to any one sensing modality.

For MIR specifically, Ng et al. (2019) reported convolutional networks outperforming PLSR on a large global library. Our results are consistent with that finding — every nonlinear method beat PLSR — but not with the implication that architectural sophistication is where the remaining gains lie. Library size is likely relevant: Ng et al. worked with tens of thousands of samples, where deep architectures have most scope to exploit capacity, whereas 1,310 samples is the scale typical of a national laboratory building its own library. At that scale, well-regularised conventional learners are difficult to beat.

### 3.10 Practical implications

Four recommendations follow directly, in the order we would prioritise them. **Sample more landscapes before refining the model**: the effect of calibration breadth measured here exceeds the spread between model families by roughly an order of magnitude. **Report the validation design alongside every accuracy figure**, stating the grouping variable, number of groups and folds, and fold-level dispersion. **Retire PLSR as the default**, since gradient boosting is a drop-in replacement requiring modest tuning and was the most reliable performer across designs. **Do not rely on spectra alone for phosphorus**, but pair the library with a small number of local wet-chemistry determinations in each new landscape.

### 3.11 Limitations

**(i)** The leave-country-out design is unbalanced in site count and its folds differ in difficulty; this is treated in full in Section 3.4, and country-level results should be read as indicative rather than as a ranking of national difficulty. **(ii)** Hyperparameters were selected on a single reserved fold rather than by fully nested cross-validation, a compute-driven compromise; the tuning fold is never used for testing, but selection was informed by data appearing in later training folds. **(iii)** The ancillary branch currently carries only sampling depth and position. An Earth Engine script accompanying this paper extracts a full satellite, terrain and climate covariate stack at the sample coordinates, but those covariates are not yet incorporated, so multi-modal fusion here means spectral plus ancillary rather than spectral plus satellite. Integrating that stack is the immediate next step and is expected to strengthen the ancillary branch materially. **(iv)** Critical levels used for the agronomic assessment are indicative reference values rather than locally calibrated thresholds, and the Olsen extraction applied uniformly here is not the appropriate convention for all soils in the panel; Section 3.8 sets out the consequences and the remedies. **(v)** All spectra derive from a single instrument in one laboratory, so instrument-to-instrument transfer (Gamagedara et al., 2025) is not tested. **(vi)** All models are the authors' own NumPy implementations, verified against analytic gradients and on synthetic data with known structure, but not identical to reference library implementations. **(vii)** Sampling took place in 2009–2013, so absolute nutrient concentrations may have shifted, although the modelling relationships are unlikely to have changed materially.

## 4.0 Conclusion

This study designed and rigorously evaluated an attention-weighted hybrid CNN–DAFM–RBF architecture for soil nutrient prediction from mid-infrared spectra, using 1,310 georeferenced samples from 16 AfSIS sentinel sites across Kenya, Ethiopia, Tanzania and Uganda.

The architecture achieves its principal design objective. A differentiable radial basis function head permits a kernel-style regressor to be trained jointly with a convolutional feature extractor and an attention fusion module, removing the separation that conventional quadratic-programming solvers impose. Attention operating over spectral positions yields weights that localise onto chemically interpretable absorbance regions, and a learned modality gate quantifies the spectral contribution as ranging from 35.0% for exchangeable calcium to 59.2% for total nitrogen — a measured account of multi-modal fusion rather than an assumed one.

On predictive accuracy the results are mixed and are reported as such. Within landscapes the architecture ranked fifth of six, with a multilayer perceptron strongest overall; under transfer it ranked mid-pack, with gradient boosting the most reliable across designs. It did, however, achieve the highest mean R² of all six models when Kenya — the primary area of application — was withheld in full.

The wider findings are of more consequence than the ranking. Predictive skill declines systematically as the withheld spatial unit widens, from mean R² of 0.749 under leave-cluster-out to 0.351 under leave-site-out and 0.205 under leave-country-out, so any reported accuracy for a soil spectral calibration is uninterpretable without its validation design. And the breadth of the calibration matters more than the choice of learner: restricting the analysis to three Kenyan sites produced negative mean R² for every model, whereas the four-country calibration achieved 0.351 and moved Olsen phosphorus from 0.164 to 0.562.

Finally, the properties most often below agronomic critical levels are not those the models predict best. Phosphorus is deficient in 84% of topsoils yet essentially untransferable between landscapes. Closing that specific gap — through targeted local calibration, complementary sensing, or extraction-specific modelling — would contribute more to fertiliser advice in East Africa than further gains on the properties already predicted well. Integrating the satellite and terrain covariate stack into the ancillary branch, and extending the calibration to additional AfSIS countries, are the immediate priorities for subsequent work.

## 5.0 Acknowledgements

The authors thank the Department of Computing at Jomo Kenyatta University of Agriculture and Technology for institutional support. We gratefully acknowledge World Agroforestry (ICRAF), Rothamsted Research and the Africa Soil Information Service for making the soil spectral and wet-chemistry data openly available, and the Bill and Melinda Gates Foundation, which funded AfSIS Phase I. We also thank the reviewers of an earlier version of this manuscript, whose detailed criticism prompted the methodological reconstruction reported here.

## 6.0 Ethical Statement

This study used secondary, anonymised soil measurement data obtained from open-access repositories under their published licences. No human participants or animal subjects were involved, and no personally identifying information was accessed or generated, so formal ethical approval was not required. The authors declare no competing financial or non-financial interests.

## 7.0 Data and Code Availability

Both source datasets are openly licensed and cited below with persistent identifiers. The complete analysis pipeline accompanies this paper and comprises model implementations, validation designs, preprocessing and metrics, dataset assembly, the experiment driver, supporting analyses, figure and table generation, an automated verification script that checks every numeric claim in this manuscript against the computed results, an Earth Engine script for covariate extraction at the sample coordinates, and a Google Colab notebook reproducing the pipeline end to end. Also released are the random seed (20260804), fold assignments for all three designs, the selected hyperparameter configurations, and per-fold predictions and metrics for all 1,794 model–property–design–fold combinations, together with raw training logs and the machine-readable result files from which every table and figure is generated. The pipeline requires only Python 3.10, NumPy, pandas and Matplotlib. All materials are available at https://github.com/jmsmuigai/Adaptive-Multi-Sensor-Fusion-for-Agronomic-Prediction

## 8.0 References

Gamagedara, D. N., Sanderman, J., Dangal, S. R. S. and Wills, S. (2025). Application of calibration transfer techniques between different mid-infrared spectrometers/modules to improve accuracy in estimating soil properties. *Soil Science Society of America Journal*, **89**(1), e70147.
https://doi.org/10.1002/saj2.70147

Guo, W., Gao, S., Ding, Y. and Dong, D. (2025). Simultaneous prediction of multiple soil components using mid-infrared spectroscopy and the GADF-Swin Transformer model. *Computers and Electronics in Agriculture*, **237**.
https://www.sciencedirect.com/science/article/pii/S0168169925006131

Hengl, T., Leenaars, J. G. B., Shepherd, K. D., Walsh, M. G., Heuvelink, G. B. M., Mamo, T., Tilahun, H., Berkhout, E., Cooper, M., Fegraus, E., Wheeler, I. and Kwabena, N. A. (2017). Soil nutrient maps of sub-Saharan Africa: assessment of soil nutrient content at 250 m spatial resolution using machine learning. *Nutrient Cycling in Agroecosystems*, **109**(1), pp. 77–102.
https://doi.org/10.1007/s10705-017-9870-x

Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.
https://arxiv.org/abs/1412.6980

Ng, W., Minasny, B., Montazerolghaem, M., Padarian, J., Ferguson, R., Bailey, S. and McBratney, A. B. (2019). Convolutional neural network for simultaneous prediction of several soil properties using visible/near-infrared, mid-infrared, and their combined spectra. *Geoderma*, **352**, pp. 251–267.
https://doi.org/10.1016/j.geoderma.2019.06.016

Padarian, J., Minasny, B. and McBratney, A. B. (2019). Using deep learning for digital soil mapping. *SOIL*, **5**(1), pp. 79–89.
https://doi.org/10.5194/soil-5-79-2019

Peng, J. (2026). A novel self-attention mechanism-based dynamic ensemble model for soil hyperspectral prediction. *Sensors*, **26**(1), Article 195.
https://doi.org/10.3390/s26010195

Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N., Rossi, V., Dormann, C., Cornu, G., Viennois, G., Bayol, N., Lyapustin, A., Gourlet-Fleury, S. and Pélissier, R. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, **11**, Article 4540.
https://doi.org/10.1038/s41467-020-18321-y

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F. and Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, **40**(8), pp. 913–929.
https://doi.org/10.1111/ecog.02881

Shepherd, K. D. and Walsh, M. G. (2002). Development of reflectance spectral libraries for characterization of soil properties. *Soil Science Society of America Journal*, **66**(3), pp. 988–998.
https://doi.org/10.2136/sssaj2002.9880

Shuai, L., Li, Z., Chen, Z., Luo, D. and Mu, J. (2025). Application of an ECAM-ConvNeXt model with multichannel spectrogram based on Vis–NIR for soil property prediction. *Journal of Chemometrics*, **39**(9), e70092.
https://doi.org/10.1002/cem.70092

Vågen, T.-G., Shepherd, K. D., Walsh, M. G., Winowiecki, L., Desta, L. T. and Tondoh, J. E. (2010). *AfSIS Technical Specifications: Soil Health Surveillance*. Africa Soil Information Service, World Agroforestry Centre, Nairobi.
https://worldagroforestry.org/output/afsis-technical-specifications-soil-health-surveillance

Vågen, T.-G., Winowiecki, L. A., Tondoh, J. E., Desta, L. T. and Gumbricht, T. (2016). Mapping of soil properties and land degradation risk in Africa using MODIS reflectance. *Geoderma*, **263**, pp. 216–225.
https://doi.org/10.1016/j.geoderma.2015.06.023

Vågen, T.-G., Winowiecki, L. A., Desta, L., Tondoh, E. J., Weullow, E., Shepherd, K. and Sila, A. (2020). *Mid-Infrared Spectra (MIRS) from ICRAF Soil and Plant Spectroscopy Laboratory: Africa Soil Information Service (AfSIS) Phase I 2009–2013*. World Agroforestry (ICRAF), V1.
https://doi.org/10.34725/DVN/QXCWP1

Vågen, T.-G., Winowiecki, L. A., Desta, L., Tondoh, J., Weullow, E., Shepherd, K., Sila, A., Dunham, S. J., Hernández-Allica, J., Carter, J., Durenkamp, M. and McGrath, S. P. (2021). *Wet chemistry data for a subset of AfSIS: Phase I archived soil samples*. World Agroforestry (ICRAF), V2.
https://doi.org/10.34725/DVN/66BFOB
