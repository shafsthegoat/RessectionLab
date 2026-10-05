# Prepared single-candidate execution wrapper

The separate [caller](../scripts/mechanics_patient_mesh_candidate_run.py) makes the [fixed graded candidate](mechanics-patient-mesh-candidate.md) runnable after a new root execution release. This is preparation only. Neither this work nor its mocked tests decode patient/source-surface arrays, read B/V landmarks, initialize a native mesher, execute a solver or update a model.

The prospective output is a fresh `outputs/mechanics/resect-case4-patient-mesh-graded-v2`; existing v1 outputs remain untouched. One candidate requests 12/24 mm boundary/interior sizing, Sampling100 and a 2–24 mm transition. It retains the frozen 6,000-node / 6,000-tet research mesh-count gate and independent 2 mm / 3% geometry limits. There are no subsequent levels, retries, parameter sweeps or solver authorization.

## Exact source and anatomy ancestry

The release requires one immutable archive whose seven files match both declared hashes and exact Git-commit bytes: this caller, the frozen candidate helper, original v1 checker, both v1/v2 declarations, existing process-group supervisor and its unchanged runtime declaration. The release binds its own exact source and output directories and refuses working-directory execution as a substitute for the archive.

Inputs are the exact retained v1 `native-surface.npz` plus the original v1 mask/runtime/header bindings. The worker checks bytes before loading the saved surface and never re-extracts an MRI or mask. The prior failed-worker record must identify that same surface hash and remain a failed result. Five additional fixed context records bind independent mask QC, the provisional baseline diagnostic, the root visual decision, that prior worker and the candidate's independent review. The archive checker and candidate carry separate source hashes; Gmsh's module/library hashes and package versions remain fixed through v1 metadata. This is not full dependency-binary attestation.

Clinical validation, anatomical registration acceptance and solver authorization remain false. The known inferior/cerebellar exclusions and failed original mesh remain evidence, not repaired anatomy. A prepared release template is deliberately unauthorized; root must populate the new committed source/archive and separately release exactly one attempt.

## Resource and publication controls

The worker uses the reviewed 180-second / 3-GiB sampled process-group supervisor and one numerical thread, with VTK explicitly sequential. A narrow wrapper adds output checks to that private supervisor's numeric observer, retaining its existing failure/kill/reap behavior and restoring the original observer afterward. No shared runtime file is edited.

The entire new attempt has an 8-MiB sampled aggregate-output limit, at most 32 files and bounded directory depth. A child-local 4-MiB hard per-file `RLIMIT_FSIZE` also bounds the native stdout/stderr log. The complete returned-candidate diagnostic packet retains its independent 2-MiB limit, including coordinates, connectivity, IDs, array headers and manifest. The previously saved 2.6-MB source surface is referenced, not copied into this attempt.

Sampled aggregate checks can detect an overrun after it occurs; they are not an atomic disk quota. The hard file limit, packet limit and final checks provide additional bounds. Normal atomic file publication may rename a listed temporary file before the observer stats it; that disappeared entry is handled as a sampling race and checked under its new name on the next/final observation. Symlinks and nonregular output files are rejected. The final acceptance receipt itself is included in a post-publication cap check; exceeding a cap can never yield success.

An interrupted worker does not claim zero native generations merely because it has not received the helper's result: its count is unknown while the helper runs, and the candidate receipt records the charge immediately before generation. Source/runtime/context hashes are checked again after execution. Parent acceptance requires successful supervision, a geometry-only result, exactly one reported generation, zero solver calls, complete diagnostic metadata, unchanged candidate-output hashes and all final resource/input gates. Even a successful result explicitly provides no solver authorization.

The prepared wrapper needs its own independent source/mocked review and incremental commit before any archive/release is created. No execution is currently authorized by the checked-in template.
