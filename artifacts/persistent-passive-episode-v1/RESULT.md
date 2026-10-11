# Persistent passive episode: generated continuity control

The reviewed task is integrated at `src/resectionlab/persistent_passive.py`.
All **115** focused canonical native, geometry, desktop and passive controls
pass in 7.21 seconds. The canonical test removes only the staging import path
from the author's 13 passing controls. Existing native stroke behavior remains.

The [saved paired episode](paired-proof.json) completes five explicit actions:
move into the cavity, acquire generated status, withdraw the moving tool,
withdraw the second tool, then STOP. Its six frames preserve full tool poses and
the generated clock. Both instruments remain represented at their access poses
after withdrawal. Native tissue/contact state is unchanged, with no removed or
contacted cells credited. The factory and its fresh replay/export take 0.450 s
in this one execution; observed worker peak RSS is 67,125,248 bytes. This is not
a comparative latency benchmark, and the four-second scenario clock is generated.

The fixture declares a pre-existing analytic 3×3 mm empty tunnel. At depth
6.5 mm, a full 6 mm shaft and tip fit within the void. The episode creates no
cavity and observes no patient. The initial second instrument is preset;
its insertion is not simulated. Fixed-axis passive motion uses existing full-tool
geometry. The independent checker covers tissue/access; pair clearance relies
on the primary checker. Fresh replay uses the same task implementation.

Review corrected a missing second withdrawal, collision checking against the
other tool's returned pose, cancellation after a committed action, and a positive
fixture that initially stopped outside the tissue. Final frames explicitly
distinguish a retained pose from a recorded access pose, without claiming
physical absence. [Independent source review](independent-source-review.json)
binds the promoted module and saved tests.

This is partial Package 2 from the hybrid steering. Generated inspection changes
visible status and STOP reason, but not decisions. No search, policy training,
patient admission, desktop replay or workspace-save integration is implied.
No mechanics, sensor, neurological or clinical claim follows. The separate
[desktop integration handoff](DESKTOP-SEAM.md) records the remaining consumer
changes; the legacy automatic-withdrawal replay must not depict this episode.

See [API and assumptions](../../docs/persistent-passive-episode.md) for execution.
