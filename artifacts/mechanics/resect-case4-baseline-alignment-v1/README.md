# Prepared Case4 baseline alignment check

**Prepared and unrun.** No MRI-before coordinates or image arrays have been
opened in this slice. Execution requires a root release binding the exact
declaration, script and coordinate-justification hashes. Limit: 115 seconds,
2 GiB sampled combined RSS, one computational thread, no automatic retry.

The allowlist contains only original T1, FLAIR, before-US and
`Case4-MRI-beforeUS.tag`. The before/during tag and during-US image are blocked
from file access; B motion and V outcomes remain closed. The imported landmark
helper and its core dependency come from isolated committed `6766a2a` sources.
Provider MD5, sizes, SHA256, original affine and script/source hashes are bound
before and after the run. Originals are never modified or resampled to new
NIfTI volumes; only small two-dimensional display samples are interpolated.
The fixed release controls are authenticated before following declared paths;
the helper manifest is authenticated before its entries. The access hook
denies other data files, symlink aliases and writes to allowed originals.
Missing, failed or timed-out memory measurements terminate the worker and
retain a failure receipt.

Coordinate semantics are justified separately in `coordinate-justification.json`:
the creator states that paired tag coordinates share the NIfTI world; the
official NIfTI definition specifies right/anterior/superior axes. The methods
identify FLAIR as MRI landmark reference. This supports identity conversion of
the declared world convention, not an identity anatomical registration.

The fixed diagnostic fits one proper, unweighted, all-point rigid transform
from FLAIR world to before-US world, without scaling, reflection, rejected
outliers or optimization using images. It reports every raw, fitted and
leave-one-out residual in mm. Each leave-one-out fit excludes that baseline
pair; these remain within-case baseline diagnostics, not the sealed V outcomes.
Poor results remain recorded without another fit. Six fixed native-plane views
compare T1/FLAIR world placement and raw/fitted FLAIR versus before-US.

**Engineering interpretation is frozen:** invalid hashes, nonfinite arrays,
degenerate points, improper transforms or reference points outside their own
image grids fail this slice. Numerical least-squares consistency is checked.
No clinical millimetre threshold is invented. Source bounds and small residuals
do not establish anatomical accuracy. Visual inspection will record gross
orientation/FOV/mismatch and uncertainty. No automatic anatomical acceptance,
brain-domain creation, cortical approval or material inference follows.

This is preparation for **retrospective intraoperative updating**. The later
manual before/during motion inputs are not available from preoperative MRI
alone; automatic motion extraction, observation timing and uncertainty remain
future work. Neither these baseline measurements nor future withheld motion
may become hidden policy inputs.

Eighteen small independent and owner checks passed in 0.39 seconds. They cover
proper rigid recovery, reflection handling, rank rejection, held-pair isolation,
physical slice sampling, control authentication, read-only access and failed
resource monitoring. Five initial execution-boundary failures are preserved in
the independent review receipt; the fit was unchanged by their repair. These
are numerical unit controls, not synthetic patient training/evaluation.
`verification.json` binds the exact tested source. Patient execution and visual
interpretation remain pending.
