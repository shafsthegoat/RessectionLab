# Offer actual residual blockers before spending the bounded candidate budget

The complete 48-step search comparison found a concrete omission: a remaining
first blocker was excluded because a wide tool could theoretically cover its
cross-section, although no wide-tool move was legal. Most offered alternatives
also failed the full-tool aperture test.

The opt-in `residual_aperture_v2` rule admits retained first blockers without
that optimistic eligibility gate. It filters only tool/cell pairs that clearly
cannot fit through the public aperture before consuming the existing 16-cell
budget. Uncertain boundaries proceed to native verification. Every emitted
candidate still requires a fresh full-tool certificate. Existing defaults,
identities, rewards, geometry and the 120+32 candidate bound remain unchanged.

Canonical validation passes 119 tests in 5.64 s. A separate generated helper
test passes in 0.32 s: a formerly omitted preparation removes a blocker, makes
the original target ray feasible, and preserves the old authority. Deterministic
native replay is checked separately from the existing independent source-cell
geometry checker. Earlier fixture failures remain retained; they changed no
production code. The archived predecessor can be used for the optional exact
default-parity test via `RESIDUAL_BLOCKER_BASELINE_DIR`.

One acquired diagnostic follows on the preselected old 47-action patient state.
It reconstructs that prefix under its original rules, offers a separately bound
prospective catalog, verifies all offered extras and tests at most three
preparations. Full prefix-plus-preparation histories receive independent
geometry checks. Original rays are re-previewed after each preparation; this
does not claim a complete accepted route or that the new rule generated the
historical prefix. Negative removal costs and the root STOP alternative remain
explicit. This is geometric simulated interaction on real anatomy, not measured
forces or neurological safety.

The run is limited to 210 s worker / 240 s parent, 10,000 previews, 60 commits
plus STOP calls, three branches and 3 GiB sampled memory. The shared storage
reservation is 32 MiB: 16 output, 8 supervision and 8 preparation. No training,
checkpoint load, new patient, acquisition or dependency is included. Source
and helper evidence are pinned in `preservation-index.json`; actual improvement
remains unmeasured until the diagnostic completes.
