**Comparison of a Hybrid Deep Learning Model with Machine Learning Methods for Mid-Infrared Soil Nutrient Prediction under Spatial Transfer in East Africa**

[Author names, affiliations and the corresponding author's e-mail address have been removed for blind review.]

```{=openxml}
<w:p><w:pPr><w:spacing w:before="7200"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">Running title: </w:t></w:r><w:r><w:t>Spectral soil nutrient prediction under spatial transfer</w:t></w:r></w:p>
```


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```


**Abstract**

Mid-infrared (MIR) spectroscopy can lower the cost of soil testing, but the reported accuracy of spectral models depends strongly on how they are validated. We compared an attention-weighted hybrid of a one-dimensional convolutional neural network (CNN) and a radial basis function (RBF) head with twelve machine learning benchmarks, using 1,310 Africa Soil Information Service samples from 16 sites in Kenya, Ethiopia, Tanzania and Uganda. Thirteen soil properties were modelled under five validation designs that withheld random samples, clusters, sites, agro-ecological zones or countries. Hyperparameters were tuned in nested, group-aware loops, neural models were trained from five random initialisations, and uncertainty was quantified with group bootstrap intervals and paired tests. Skill fell as the withheld unit widened: the median model's mean coefficient of determination (R²) declined from [[median_model_random:.2f]] under random splitting to [[median_model_site:.2f]] for a new site and [[median_model_country:.2f]] for a new country. Gaussian process regression ranked first under every transfer design (leave-site-out mean R² [[best_site_r2:.2f]]), although eleven benchmarks were statistically indistinguishable from it. The hybrid ranked last (mean R² [[hyb_site_r2:.2f]]). Ablation showed that positional information and attention pooling were essential, whereas the RBF head and the modality gate gave no benefit, and the RBF head did not limit worst-case errors. Calibration diversity mattered more than size, and adding 50 local samples from the target site raised the mean R² of kernel ridge regression from [[sp_KRR_0:.2f]] to [[sp_KRR_50:.2f]]. Group-aware conformal intervals reached [[cf_site_group_mean:.0%]] coverage of a nominal 90% under leave-site-out, compared with [[cf_site_random_mean:.0%]] with random calibration. Organic carbon, pH and effective cation exchange capacity could be screened against critical levels in unsampled landscapes, but Olsen phosphorus remained unpredictable even with local samples. Diverse calibration libraries and honest validation offer more than added architectural complexity.

**Keywords:** mid-infrared spectroscopy; soil nutrient prediction; spatial cross-validation; ablation; conformal prediction; Gaussian process regression; Africa Soil Information Service; Kenya

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

