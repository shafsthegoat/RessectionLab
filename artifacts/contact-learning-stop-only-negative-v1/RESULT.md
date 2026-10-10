# Actual TRAIN smoke: no non-STOP proposal

The first actual learning integration smoke at `33f49a1` stopped after
4.273 seconds, 361,906,176 bytes sampled peak RSS, with clean owned termination.
The predefined TRAIN layout `pcf-06`, surface goal, produced **zero legal
non-STOP actions**. SEARCH returned STOP after zero search transitions. This
is a complete search result for the declared inventory, not evidence that all
continuous instrument motions are impossible.

The shared teacher trajectory was executed, sealed and replayed. The one-update
IL call completed, but the smoke's parameter-change criterion failed. A
categorical policy with only STOP has no action-selection gradient. No partial
checkpoint, v3 export, held-out case or full pilot followed. Loss/update details
were unfortunately assigned to the result after the assertion and were not
saved; later instrumentation must record them before checking the outcome.

Source and existing preview-ledger inspection identify the geometric issue:
the aperture center is x=5.75 mm with radius 1.5 mm, while the fixed proposal
lattice places the nearest entry at x=6 mm. The aspirator envelope radius is
1.25 mm, leaving exactly zero aperture clearance. The unchanged strict geometry
rule rejects contact with the aperture boundary. Other proposed columns fit
worse; probe actions require an existing cavity. This is a proposal coverage
limitation, not a reason to weaken collision checks or substitute another case.

A separately bounded preview will test the omitted continuous aperture-center
ray, keeping the same anatomy, tools, aperture and constraints. Any corrective
proposal representation must receive a new decision-model identity while
preserving layout recipes and patient/geometry roles. The earlier STOP-only
positive factory check and 96 passing source tests did not establish a useful
learning action inventory; this negative provides that missing integration test.
