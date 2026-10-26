**4.0 Conclusion**

This study evaluated an attention-weighted hybrid CNN–RBF model for MIR soil nutrient prediction against twelve established machine learning benchmarks. The evaluation used 1,310 AfSIS samples from 16 sentinel sites in four East African countries, five validation designs, nested group-aware tuning, five random initialisations per neural model, and group-bootstrap inference.

The proposed architecture did not achieve its design objective. It ranked last in every validation design. Ablation showed that the components that motivated it, the RBF head and the modality gate, added no skill, and that the RBF head's bounded output did not reduce worst-case errors. The parts that mattered were generic: positional information and position-aware pooling of convolutional features. The extreme failures that did occur came from back-transforming log-modelled targets, not from the output layer: they involved unbounded models, and they disappeared when predictions were limited to the training range.

The wider findings are of more practical consequence. First, validation design dominated every comparison: moving from random splitting to a new site, climatic zone or country removed about a third to two-fifths of the apparent skill, so any accuracy reported for a soil spectral library must state the spatial unit that was withheld. Second, well-tuned conventional learners were statistically close; GPR ranked first under all transfer designs but was distinguishable from few alternatives. Third, calibration diversity mattered more than calibration size: at a fixed size, more sites gave better transfer, and 20–50 local samples recovered a substantial part of the lost skill. Fourth, group-aware conformal calibration substantially improved interval coverage under transfer, although coverage stayed below the nominal 90% in every transfer design and furthest below it for unseen countries.

Organic C, pH, total N and effective CEC can be predicted in unsampled East African landscapes with RPD of at least 2, and organic C, pH and effective CEC can be screened reliably against indicative critical levels. Olsen P cannot, with or without local spiking. Phosphorus advice therefore still requires direct measurement, with the spectral library used to decide where to sample. Priorities for further work are to add sentinel sites from under-represented agro-ecological zones, to scan and analyse fresh samples together so that storage effects can be excluded, to use the unlabelled spectra for self-supervised pretraining, and to make uncertainty depend explicitly on distance from the calibration domain.

**5.0 Acknowledgements**

[Removed for blind review.] We acknowledge World Agroforestry (ICRAF), Rothamsted Research and the Africa Soil Information Service for making the soil spectral and wet-chemistry data openly available, and the Bill & Melinda Gates Foundation, which funded AfSIS Phase I. We thank the anonymous reviewer for detailed and constructive comments.

**6.0 Ethical Statement**

This study used secondary, anonymised soil measurement data obtained from open-access repositories under their published licences. No human participants or animals were involved, and no personally identifying information was accessed or generated; formal ethical approval was therefore not required. The authors declare no competing financial or non-financial interests.

**7.0 Data and Code Availability**

Both source datasets are openly licensed and are cited with persistent identifiers (Vågen *et al.*, 2020, 2021). The complete pipeline will be released in a public repository on acceptance; the link is withheld here for blind review and is available to the editor on request. The pipeline comprises dataset assembly, preprocessing, all model implementations, fold definitions, the nested tuning and seed protocol, every additional experiment, and a single script that regenerates every table, figure and number in this manuscript from the stored per-fit records. Fold assignments, selected configurations, per-seed out-of-fold predictions and the library-agreement records are included.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

**8.0 References**

[[REFERENCES]]
