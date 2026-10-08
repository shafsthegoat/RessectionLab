# Finer axial spatial convergence

The previous N8/N12 half-height equivalence result permits a narrower, cheaper
numerical refinement experiment. It does not calibrate human brain material.
The original spatial-convergence failure and full-domain N24 timeout remain
unchanged. This experiment must pass before measured-response fitting is opened.

The [frozen declaration](../manifests/experiments/hbe-01-03-halfheight-spatial-v1.json)
retains the original specimen geometry, 60 steps, 75 physical probes, numerical
gauge modulus, boundary conditions and acceptance thresholds. It reuses five
complete full-domain results and adds exactly three native half-domain solves:
compression N24, tension N16 and tension N24. Both N12–16–24 and N8–16–24
triplets must pass in both branches. Reconstructed full fields are labeled as
reflections; they are not full-domain native measurements or physical validation.

Preparation and solving have separate source-bound execution records. The full
archive contains14 modules and four declarations. No new mesher calls are needed.
Limits:60 seconds preparation;1200 seconds inclusive solve/readout allowance;
420 seconds per native call; one thread;3 GiB sampled process-family memory;
512 MiB per active run and2 GiB retained output. One final clock reading controls
both acceptance and the recorded duration. Final retained-size checking includes
the exact serialized result receipt. Receipt publication itself is outside that
clock measurement. Failure never authorizes retries or deletion of partial logs.

Independent source review and geometry/deck controls passed before execution.
Nine pure file/clock controls pass, including equality at the time limit, exact
byte-cap boundaries, leftover temporary files, symlinks and publication failures.
All11 inherited source pins are unchanged. The reviewer initially asserted the
wrong bottom-constraint representation; comparison with the original grouped
xyz constraints corrected the review, with no production-code change.

Run the archived runner with `--phase prepare` or `--phase solve`, `--root` set
to the primary checkout, and the respective release path/hash plus `--execute`.
The release binds the committed archive, interpreter and repaired FEBio runtime.
Execution results and complete raw-log review must be recorded separately.
The [executed result](../artifacts/mechanics/hbe-halfheight-spatial-execution-v1/RESULT.md)
retains a failed compression-force trend despite all three individually passing
native runs and improved displacement differences. Independent replay matches
all eight readouts. No measured response, fitting, RL or surgical-validity claim
follows from these software checks or numerical outputs.
