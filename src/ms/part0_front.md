**Does Architectural Complexity Pay Off Under Spatial Transfer? Ablation, Uncertainty and Nested Validation of an Attention-Weighted Hybrid CNN–RBF Model for Mid-Infrared Soil Nutrient Prediction in East Africa**

**Running title:** Hybrid CNN–RBF soil prediction under transfer


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```


**Abstract**

Mid-infrared (MIR) spectroscopy can lower the cost of soil testing, but the reported accuracy of spectral models depends heavily on how they are validated. We evaluated an attention-weighted hybrid model that combines a one-dimensional convolutional neural network (CNN), a Dynamic Attention Fusion Module, a modality gate and a radial basis function (RBF) head. It was compared with twelve library-implemented benchmarks on 1,310 Africa Soil Information Service samples from 16 sentinel sites in Kenya, Ethiopia, Tanzania and Uganda. Thirteen soil properties were modelled under five validation designs: random, leave-cluster-out, leave-site-out, leave-agro-ecological-zone-out and leave-country-out. Hyperparameters were selected by nested, group-aware tuning, neural models were trained from five random initialisations, and uncertainty was quantified with group bootstrap intervals and paired tests. Skill fell as the withheld unit widened: the median model's pooled R² averaged over the thirteen properties declined from [[median_model_random:.2f]] under random splitting to [[median_model_site:.2f]] for a new site and [[median_model_country:.2f]] for a new country. Gaussian process regression ranked first under all three transfer designs (leave-site-out mean R² [[best_site_r2:.2f]]), but eleven other models were statistically indistinguishable from it. The hybrid ranked last (mean R² [[hyb_site_r2:.2f]]). Ablation showed that positional information and attention pooling were essential, whereas the RBF head and the gate gave no benefit, and the RBF head did not limit worst-case errors. Calibration diversity mattered more than size: at a fixed size, training on more sites improved prediction of Kenyan sites, and adding 50 local samples from the target site raised the mean R² of kernel ridge regression from [[sp_KRR_0:.2f]] to [[sp_KRR_50:.2f]]. Group-aware conformal intervals reached [[cf_site_group_mean:.0%]] coverage of a nominal 90% under leave-site-out, compared with [[cf_site_random_mean:.0%]] with a random calibration set. Organic carbon, pH and effective cation exchange capacity were predicted well enough in unsampled landscapes to screen samples against critical levels, but Olsen phosphorus remained unpredictable (R² < 0.1) even with local spiking. Broad, diverse calibration libraries, honest validation and phosphorus-specific measurement offer more than added architectural complexity.

**Keywords:** mid-infrared spectroscopy; soil nutrient prediction; spatial cross-validation; ablation; conformal prediction; Gaussian process regression; Africa Soil Information Service; Kenya

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

