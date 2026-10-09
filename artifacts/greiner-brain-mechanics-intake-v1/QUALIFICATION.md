# Greiner specimen-response qualification

Source review, October 9, 2026. The acquired CSVs remain unparsed; no response values, fit, solver run or evaluation result are reported here.

The [primary paper](https://doi.org/10.1098/rsfs.2024.0026), sections 2.1 and 2.5 ([open full text](https://eprints.gla.ac.uk/343003/1/343003.pdf)), describes **two individual specimens from one 71-year-old male donor**: visual cortex and corona radiata. Both have 4 mm radius; heights are 3.4 and 5.0 mm respectively. Axial force is converted to nominal stress using undeformed area. The protocol applies three compression/tension cycles at 40 µm/s over stretch 0.85–1.15, followed by 300-second holds loaded at 100 µm/s. Both ends are glued, and specimens are submerged in PBS at 37 °C. Published curves are filtered and point-reduced; CSV units, columns and segment boundaries still require qualification.

These measurements can test axial constitutive response and stress relaxation. Prescribed plate motion is not an independently measured internal displacement field. The separate two-hour diameter observation concerns a different 57-year-old donor and is not established as an acquired displacement series. Neither record validates live surgical retraction, cutting, tool contact or neurological harm.

The [v2 release, 13960486](https://zenodo.org/records/13960486), adds code, build information and an example parameter file. Official version metadata gives the same file IDs, sizes and MD5 hashes for both CSVs as the [acquired v1 release](https://zenodo.org/records/13946193). This is no additional donor or corrected response dataset. V2 code and parameter bodies have not been acquired in this qualification. HBE donor/specimen overlap remains unresolved; common investigators and methods cannot establish independence.

## Next bounded experiment

After schema and loading-history checks, prospectively select the corona-radiata specimen, fit only declared cyclic segments, freeze the model and predict the compression hold. Keep the tension hold and visual-cortex responses sealed. Evaluate an established FEBio viscoelastic option against an elastic/no-relaxation baseline, with explicit geometry, boundary, constitutive and numerical assumptions. Predeclare time-weighted nominal-stress RMSE, endpoint amplitude error and interpolation/quadrature for the point-reduced observations. Report identifiability and sensitivity: these traces do not identify every porous, viscous and volumetric parameter.

**Leakage guard:** section 3.4/Table 5 of the paper fits the three cycles **and both relaxation traces**. Its fitted parameters and the v2 tuned defaults must not become initialization, fixed constants, bounds, priors or tuning guidance for the proposed held-out relaxation test. Inspect parameter provenance separately from solver methods. No existing full-trace fit can be relabeled a withheld prediction.

This proposal could establish within-specimen prediction across loading modes. It cannot establish independent-donor transfer. No roles have yet been assigned and no scientific-use admission follows from this source review. The missing evidence for direct retraction validation remains same-trial calibrated tool/contact/load and independently registered tissue motion with uncertainty.

Independent review: `build/greiner-scientific-qualification-independent-v1/REPORT.md`. The corrected source qualification is `build/greiner-scientific-qualification-v1/qualification.json`, SHA-256 `9e4f28307f2cf7a23d73b2f402e39f03d5a977d451315ffb58fe33a4557631b5`. Original intake hashes and acquisition receipts remain in [manifest.json](manifest.json).
