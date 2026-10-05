# PAT25 ingress-access diagnostic

The single declared attempt completed. Deterministic static ingress screening selected source axis 2, outward sign −1. Its initial native inventory accepted **78/78 previews**, compared with **0/78** for the unchanged original access. This establishes availability of initial preview actions in this geometric simulation. It does not establish learned-policy improvement, complete resection, clinical benefit, or clearance for an operation.

The study used one fixed TRAIN case, PAT25, and all six prespecified source-axis exits. Selection used any admissible static entry pose, then distance, axis and sign; it did not select by subsequent preview acceptance, reward or actor coverage. No fallback selection was attempted.

| Source axis, outward sign | Existing local exit distance (mm) | Static admissible / returned poses | Disposition |
|---|---:|---:|---|
| 0, −1 | 17.5001 | 0/78 | Rejected; original baseline |
| 0, +1 | 29.5002 | 78/78 | Eligible |
| 1, −1 | 96.5000 | 66/78 | Eligible |
| 1, +1 | 25.5000 | 60/78 | Eligible |
| 2, −1 | 20.5000 | 78/78 | Eligible; chosen |
| 2, +1 | 85.5000 | 72/78 | Eligible |

All six exits were assessed: five eligible, one rejected, none with an unfinished eligibility assessment. **All 468 returned static poses retained both `tool_geometry_outside_image_unassessed` and `normal_tissue_exposure_unassessed`.** Eligibility therefore coexists with unknown geometry. Rejected static poses reported a shaft blocked by remaining native tissue. These annotation-assisted local apertures remain unreviewed; they do not establish skull or craniotomy access.

| Tool | Original accepted / emitted | Selected accepted / emitted |
|---|---:|---:|
| Fine aspiration | 0/39 | 39/39 |
| Wide aspiration | 0/39 | 39/39 |

Both native inventories completed with all 78 declared slots emitted and evaluated, no omissions or duplicate substitutions. The baseline reproduced its historical initial inventory. All 78 original full previews were rejected for a shaft blocked by remaining native tissue. Initial state invariants matched after each inventory. An accepted preview is hypothetical: no cut or transition was committed, and the result does not show that an entire sequence or the target volume can be removed.

Both actor crops retained **100% of nominal target mass** (16,526 positive source cells); there was no target-mass gain. The available initial cavity channel contained zero source and visible cells in both phases. Motor and language channels remained unavailable. The following means use all 78 emitted previews per phase, including rejected previews:

| Continuous centerline segment | Original center domain | Selected center domain | Original full-cell extent | Selected full-cell extent |
|---|---:|---:|---:|---:|
| Entry → tip | 99.7841% | 100.0000% | 99.8920% | 100.0000% |
| Approach shaft | 11.9134% | 11.9133% | 12.3389% | 12.3388% |
| Deepest shaft | 29.0769% | 24.0575% | 29.5024% | 24.4831% |

The actor's deepest shaft centerline visibility decreased despite improved preview availability. Center interpolation domains and full-cell extents are distinct; neither measures complete-instrument visibility or clearance. The original accepted-only summaries are null because their denominator is zero. The selected accepted-only summaries have denominator 78. Five-point samples and accepted-envelope summaries remain in the compact record; they are not substituted for continuous coverage or removed volume.

There were exactly **468 unique static checks**, **156 native preview calls**, and **two initial inventories**. No static poses were reused. Recorded searches, policy forwards, optimizer updates, committed transitions and checkpoints were zero. No retry occurred.

Worker elapsed time was 34.400688 s; the inner supervisor recorded 34.878337 s and the outer receipt recorded 36.754634 s. These nested intervals are not additive. Recorded phase times were 5.258874 s for candidate setup, 6.489613 s for screening, 2.812906 s for the original inventory and 12.004727 s for the selected inventory. They are descriptive costs, not an isolated speed comparison. The outer sampled process-group RSS peaked at 2,609,086,464 bytes (2,488.21875 MiB; 292 samples), below the declared 6 GiB threshold. Sampling can miss transient peaks. The outer receipt recorded exit 0 and a closed process group, within the 180 s whole-attempt limit; its clock excludes interpreter startup before its first timestamp and OS scheduling.

The [independent saved-only audit](independent-audit-initial.json) passed on its first attempt. It checked six-exit selection, saved inventories, coverage arithmetic, source/release joins, 75 bound inputs and all seven raw output files without changing them. Its fixed tolerances were 1e−12 for scalar arithmetic and 1e−8 mm for physical coordinates. It did not reload dense patient arrays or independently recompute occupied cells, crop integrals or clearance, and it does not establish clinical safety or efficacy.

The [compact summary](compact-summary.json) and [comparison figure](ingress-comparison.png) retain the fixed comparisons and negative findings. [Formatting provenance](plot-formatting-provenance.json) preserves the initial plot-helper bytes/hash and initial render hashes; the sole rendering correction separated the panel note from the footer. The final figure was visually inspected. Raw evidence and its lossless archive were retained separately by the parent task; reporting performed no patient decoding, native rerun or refitting.
