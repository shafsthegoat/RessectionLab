# Independent native spatial task and real-patient preflight review

The final focused run passed **55 tests in 3.58 seconds**, including 22 new
independent checks. All executed transitions were fixed analytic geometry
controls. No patient images or episodes, policy training, optimizer update,
public preflight, or synthetic learning benchmark was executed by this review.

The independent geometry fixture has six occupied source cells and two fixed
tools on a rotated, reflected 24×25×18 native grid. The actor sees a 5³ crop.
Tests enumerate all 12 declared cell/tool pairs, retain infeasibility reasons,
and match projected entry and tip coordinates to actual policy action rows.
Two prescribed cuts remove source cells beyond the actor crop while the native
grid and independent full-history checker remain authoritative. Partial-contact
cells remain occupied, their count is distinct from removal, and their objective
weight is explicitly zero. Motor/language costs are disabled and outcomes remain
unassessed/null. No clinical accuracy claim follows from this fixture.

Changing only the evaluator's private target alters rewards but leaves candidate
order, masks, IDs, observed tensors, decision-model identity and the permitted
nominal-target planning clone unchanged, both before and after a cut. Tool,
access, horizon and objective mutations fail closed. Nonzero scenario seeds
are refused: this is one deterministic structural geometry world.

Initial runner review found two executable authority gaps. A caller could reseal
cohort metadata to promote PAT26 or PAT29 to training, and supply a README-only
source inventory. Both reached an instrumented image-loader boundary in the
archived initial runner. The loader raised before reading any image bytes.
The counterexamples and initial source are preserved here. Static review also
found that the parent passed a mutable declaration path without binding its
bytes to the worker.

The owner repaired these gates. The runner pins the original cohort path and
SHA, accepts only the six original/additional TRAIN IDs with their expected
roles, requires the runner and complete headless Python source inventory, and
checks these before bundle checksum/loading. Parent and worker now share a
saved exact-byte declaration and SHA. Independent tests reject altered case,
tool, access and settings bytes, resealed role promotions and omitted sources.
The valid TRAIN metadata path remains callable without opening any images.

`receipt.json` records final source hashes and test logs. All ten captured tested
source files remained unchanged during the final run. Root retains responsibility
for the committed immutable source and any real-patient release. The patient
support/access declaration was not created or approved by this review.

Reproduce the focused checks:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_native_spatial_task_review.py tests/test_native_spatial_task.py tests/test_real_spatial_preflight.py
```

This is a structural pipeline prerequisite. Candidate completeness is limited
to the declared primitive family; the crop is a partial view. The adapter does
not yet implement probabilistic motor/language planning, and a one-update
preflight would establish integration rather than useful learning.
