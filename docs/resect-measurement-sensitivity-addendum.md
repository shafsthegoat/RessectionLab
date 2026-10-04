# RESECT measurement scales and prospective sensitivity addendum

Research-only proposal, 2026-10-04; root review and a separate execution declaration required. This note does not modify the frozen [Case4 contract](resect-conditional-displacement-poc.md). No local landmark text, destination measurements, image arrays or during-US image were read for this review. Published aggregate case summaries were already consulted during acquisition research; Case4 remains DEVELOPMENT.

## Primary evidence and its scope

The [creator manuscript](https://users.encs.concordia.ca/~hrivaz/Xiao_Database_MedPhys.pdf) reports:

| Location (PDF pages, one-based) | Observation |
| --- | --- |
| §II.B, pp.6–7 | Most MRI voxels are 1 mm isotropic; five glued fiducials support image-to-patient registration. |
| §II.C, p.7 | Factory-calibrated probes use Polaris optical tracking; reconstructed US spacing is 0.14–0.24 mm isotropic. No dataset-specific tracking/calibration uncertainty is supplied. |
| §II.F, pp.13–14 | Reference points stay fixed. Two raters each repeat destination picking twice, 1–2 weeks apart; released destinations average four picks. |
| Table III, p.14, before/during row | Within-rater mean distances: 0.39±0.11 and 0.31±0.07 mm; between-rater averages: 0.27±0.05 mm. These summarize patient-level mean Euclidean distances. |
| §III, p.14 | T1 is rigidly registered to FLAIR; no numerical registration-uncertainty estimate is provided. |

Thus repeatability excludes repeated reference picking and acquisition/tracking variation. Neither voxel spacing nor repeatability establishes localization accuracy. Shared frame bias, correlated annotations and registration error remain unresolved. The paper supplies no joint covariance; converting these radial-distance summaries into axis variances, adding them in quadrature, or dividing by √4 would introduce unsupported assumptions.

The [creator's RESECT instructions](https://yimingxiao.weebly.com/data-repositories.html), “Instruction for RESECT dataset,” specify world millimetres, reference MRI/before-US first, and corresponding MINC/NIfTI world coordinates. This establishes coordinate interpretation, not physical accuracy.

## Proposed bounded stress test, not a noise model

The following are engineering choices motivated by the published scale, not estimated Case4 uncertainties. Freeze them, all model settings and output hashes before V access. Keep role IDs, support radius and original observations immutable. Run only after the nominal numerical gates pass.

Use a fixed 0.5-mm perturbation magnitude and exactly two deterministic patterns on the six B IDs sorted ascending: common signs `(1,1,1,1,1,1)` and alternating signs `(1,-1,1,-1,1,-1)`. For each pattern, physical coordinate axis and global sign, create one altered input: **12 variants per family**.

- **Destination family:** perturb only B destination coordinates; retain source coordinates/operator. This probes a shared stage offset and one differential annotation pattern.
- **Reference family:** perturb only B source coordinates; hold destinations fixed, recompute displacement and the same physical observation operator. This probes error in reference locations and observation support together.

Total: at most **24 additional solves**, no random draws, material fitting, setting selection or retries. All baselines receive the same perturbed B. Invalid support/rank/domain or solver failure remains a failed variant, without snapping, exclusion or replacement. V observations stay unchanged and evaluator-only; report prediction changes, paired ranking changes and failures for every variant. These patterns do not cover all correlations or prove robustness within a 0.5-mm ball.

A coherent coordinate transformation applied to domain, B and predictions should preserve physical results; test that separately as numerical equivariance. It cannot measure shared tracking bias. Differential MRI/US registration sensitivity needs a separately frozen transform-perturbation experiment; baseline fit/leave-one-out residuals alone do not calibrate it.

## Proposed numerical acceptance budget

At the three prospectively fixed mesh scales, require medium-to-fine prediction change ≤0.05 mm RMS and ≤0.10 mm maximum. Require physical averaged-constraint residual ≤0.01 mm and a single predeclared tighter-solver/finer-load-step control changing predictions by ≤0.01 mm maximum. Freeze identical physical probe positions from the permitted domain and B inputs alone; V positions must not determine numerical stopping or mesh refinement. Report quadrature/interpolation errors separately and retain positive-Jacobian, observation-rank and convergence checks. Budget failure blocks a numerical-resolution claim; no V-directed refinement or tuning follows.

These thresholds allocate computation below the reported picking scale; they are proposed engineering gates, not published error bounds. Three meshes and one tighter control cannot prove asymptotic convergence or bound all numerical error. Wall-time/memory caps require a separate source-bound declaration because measurement resolution supplies no runtime estimate.

## Claims this can support

Report conditional retained-landmark errors in millimetres and sensitivity across the named variants. No calibrated confidence interval, patient-specific total covariance, absolute application tolerance, dense-field accuracy, force/injury validation or clinical safety claim follows. An application tolerance requires an independently justified intended task, spatial coverage, failure policy and measurement/calibration study. The existing research cannot supply that tolerance merely by choosing a multiple of voxel spacing or repeatability.
