# Evidence for the next surgical RL simulation slice

Checked October 4, 2026. These primary studies and clinical guidelines constrain
model design; they do not validate RessectionLab. Implementation implications
below are our engineering interpretation. No experiment or dependency change
was performed for this review.

1. **Make MRI availability explicit.** EANO specifies T2, FLAIR, and 3D T1
   before/after gadolinium as the diagnostic structural baseline. Perfusion and
   other advanced imaging serve additional purposes. Do not require CT as the
   default glioma input, or treat an available CT as equivalent to these MRI
   sequences. Store acquired sequences, dates, physical frames, coverage and
   preprocessing separately; absence of a sequence must remain visible.
   [EANO guideline, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7904519/).

2. **A public dataset is not a complete surgical record.** UCSF-PDGM describes
   500 preoperative MRI cases, diffusion/perfusion imaging, and molecular
   metadata from histopathologically proven gliomas. This does not establish
   that every downloaded derivative contains every acquisition, intraoperative
   observation or mechanics measurement. Segmentation is a derived annotation:
   retain its compartments, reviewer and source sequence. Do not turn FLAIR
   abnormality into histologically certain tumor or the remaining brain into
   verified normal tissue. Molecular availability must be checked against the
   planning cutoff, independently of the MRI date.
   [Dataset authors, 2022](https://arxiv.org/abs/2109.00356).

3. **Functional imaging needs patient evidence and validation.** A prospective
   26-patient study compared task fMRI and diffusion tractography with direct
   cortical stimulation and corticocortical evoked potentials. Performance
   depended on the reference test; concordant and discordant modalities behaved
   differently. Therefore, preserve task, acquisition/reconstruction settings,
   stimulation site and mapping context. Missing or negative imaging cannot
   certify functional absence. Population atlases, tract proposals and actual
   intraoperative responses must remain distinct; published group performance
   must not become this patient's deficit probability.
   [Language-mapping validation, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10248363/).

4. **Preoperative geometry becomes stale.** Serial intraoperative MRI in 25
   patients measured deformation at multiple surgical stages, including surface
   shift and subsurface changes with cavity collapse. A rigid preoperative
   clearance certificate cannot establish clearance after tissue movement.
   Future intraoperative MRI/ultrasound, mapping and monitoring observations need
   timestamps, registration uncertainty and explicit state updates. For current
   offline RL, freeze a declared deformation scenario for each world; a newly
   observed surgical state requires a new evaluated state/version, not silent
   reward or anatomy changes during optimization.
   [Nabavi et al., 2001](https://pubmed.ncbi.nlm.nih.gov/11322439/).

5. **Resection, ischemia and function are separate outcomes.** Current EANS–EANO
   guidance assesses resection with postoperative MRI, distinguishes enhancing
   and nonenhancing residual volumes, and uses DWI to help distinguish ischemia.
   Neurological and neurocognitive outcomes require their own assessment.
   Report accessible target, modeled removal, residual annotation, non-target
   removal, contact and functional exposure separately. A weighted RL return
   is a research preference; geometric overlap is not an injury measurement,
   and simulated removal is not MRI-confirmed surgical extent of resection.
   [EANS–EANO guideline, 2026](https://academic.oup.com/neuro-oncology/article/28/1/38/8256732).

6. **Retraction needs a calibrated material model.** Human-brain experiments
   measured shear, compression, tension and relaxation across four regions.
   Their viscoelastic models capture rate/history dependence, hysteresis and
   tension–compression asymmetry. A structural intensity or binary mask does
   not supply these material parameters. A mechanics slice needs declared
   material distributions, loading rates, contact/friction, skull/dura/CSF
   boundary assumptions and time integration. Validate predicted displacement
   and force on held-out measured loading histories before using mechanical
   penalties as more than experimental surrogates.
   [Budday et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28658600/).

7. **Tool dimensions establish geometry, not safe cutting.** Human-cadaver
   measurements differed by anatomical region and maneuver, with explicit
   limitations from sensor resolution, cadaver properties and operator/tool
   coverage. General tip, shaft, length and angle dimensions can support a
   declared swept-envelope collision calculation. They cannot establish cutting
   efficiency, acceptable retraction force, living-tissue injury thresholds or
   commercial-device equivalence. Model insertion, blunt displacement, sharp
   cutting, suction and energy delivery as different actions when implemented;
   each needs its own measured response. Do not transfer cadaver force averages
   into clinical safety limits.
   [Microneurosurgical force experiment, 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4377085/).

8. **Use surgical RL frameworks for task design, with scoped validation.**
   LapGym provides SOFA-based deformable manipulation/dissection environments
   and PPO baselines; SurRoL supplies ten dVRK-oriented tasks and reports robot
   transfer. These support contact-rich observations, action constraints and
   reproducible benchmarks, not glioma outcome calibration. Validate any adopted
   component for our tissue, tools and observation model.
   [LapGym, 2023](https://jmlr.org/papers/v24/23-0207.html),
   [SurRoL, 2021](https://arxiv.org/abs/2108.13035).

9. **Cutting requires more than deleting contacted cells.** DiSECt combines FEM,
   contact and damage evolution, calibrates against measured force/deformation,
   and tests physical knife motions on fruit. It demonstrates a calibration
   approach, not validated brain cutting. A brain-cutting claim would require
   matched tool trajectories, forces, specimen conditions, deformation and
   separation/removal observations across held-out speeds and specimens. Injury
   prediction additionally needs biological outcomes; force fit alone is
   insufficient.
   [DiSECt extended study, 2022](https://arxiv.org/abs/2203.10263).

**Immediate implementation consequence:** retain the current engine as a
geometric removal/contact simulator. Improve maneuver/state semantics before
claiming mechanics: entry and working-space constraints, explicit activation,
incremental cavity changes, persistent tool pose and collision-checked withdrawal/STOP.
Benchmark search and scratch/adapted policies on the identical frozen action,
observation and uncertainty contracts. Add a separately labeled, calibrated
mechanics experiment only when its measurement data exist; do not assign
fabricated forces or injury probabilities to make the simulation appear richer.
