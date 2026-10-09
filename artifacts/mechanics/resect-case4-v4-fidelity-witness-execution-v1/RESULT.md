# Case4 rejected-mesh fidelity witness: one private diagnostic

The one source-bound saved-geometry diagnostic completed and exactly replayed
the rejected v4 mesh's original 1 mm bidirectional distance test. Both 2 mm
surface gates still fail: sampled maxima are 3.869 mm source-to-mesh and
2.264 mm mesh-to-source, with full-surface upper bounds of 4.576 and
3.254 mm. The v4 mesh remains rejected; no mesher, solver, image, landmark,
policy or training operation ran in this attempt.

The private per-triangle audit found samples over 2 mm on 535 of 183,902
source faces and 5 of 4,856 mesh-boundary faces. Those whole faces account
for 0.2879% and 0.1313% of the two surface areas, in 16 and 5 connected
groups. Boundary corners stayed on the source surface within numerical
precision; no straight midside exceeded 2 mm, while two mesh-face centroids
did (maximum 2.315 mm). These are geometric observations about the retained
estimated T1 envelope, not anatomical or clinical measurements.

The cube selected by the prospectively fixed 30 mm / 5 mm grid rule captured
25.25% and 45.52% of the two directions' exceeding whole-face area under the
max-min rule, below the required 80% in each direction. The **single local
refinement premise was falsified at that fixed scale**. This does not prove
that every possible targeted meshing method would fail; it gives no warrant
for another Case4 mesh run now. The separate anatomy/frame/support and
sparse-solver gates remain open.

The worker made exactly 1,193,462 declared distance/probe queries and finished
in 3.48 supervised seconds at 260 MB sampled peak process-group RSS, with
private output under 2.2 MB. The [sanitized summary](summary.json) records
aggregate metrics and hashes; raw coordinates and patient-derived arrays
remain in ignored private output. The
[independent saved-output audit](independent-review.md) rechecked source,
input and output bindings and recomputed the saved-array aggregates. It did
not independently rerun the complete VTK nearest-triangle oracle. Case4 is
retrospective DEVELOPMENT because its comparison outcomes were previously
exposed. No tissue displacement, force, cutting, injury or planning benefit
was validated.
