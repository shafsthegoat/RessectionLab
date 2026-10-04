# Minimal reviewable implementation after the isolated probe

The experiment supports a bounded, explicit alternative proposal inventory. It
does not justify changing geometry, broadening an existing selected route
silently, or changing an old experiment's candidate set.

1. Add a small experimental proposal module with a frozen
   `AxisColumnProposalConfig` and a pure `propose_axis_columns(config, state)`
   function. Port the tested rule from `scripts/probe_native_frontier.py` without
   changing its thirteen offsets, endpoint/fallback policy or aperture filter.
   Return exact tool/entry/target tuples, stable content-derived IDs, omitted
   proposals and explicit unsupported-access reasons. The output contains no
   clearance certificate and never alters remaining tissue.
   Harden the prototype's prechecks using `rtol=0`, explicit off-axis component
   limits and finite nonsingular affine validation. Name out-of-image omissions
   instead of silently skipping them. Regress 0.001-radian tilt and
   0.0005-voxel fractional-origin cases; neither occurred in the completed run.

2. Add an optional proposal-provider field to the native adapter in a separate
   reviewed change. The fixed-ray provider remains the default. Hash the
   provider version, parameters, proposal budget and action-limit policy into
   the decision model. Dynamic proposals depend only on visible remaining
   source tissue and supplied labels; the generator configuration stays frozen.
   Do not mutate the existing fixed `candidate_tips_mm` array during an episode.

3. Certify every proposed stroke with the existing `preview_stroke` and commit
   only its state-bound certificate. Preserve complete insertion/retraction,
   prior-cavity shaft checks, hard exclusions, full-cell containment and
   face-connected removal. Export all attempted proposals and failures. Keep
   candidate-generation/checking budgets distinct from the maximum number of
   actions offered to a policy; the existing default seven-action limit would
   otherwise truncate this larger inventory in ordering-dependent ways.

4. Reuse the existing fifteen physical action features and six state features,
   unchanged physical reward and partial-contact accounting. Exact entry and
   endpoint are part of each action identity. Bind the initial geometry cache to
   the new provider fingerprint. Give SEARCH, GREEDY, scratch, frozen and
   adapted methods the same declared provider and certification budget in any
   later comparison.

5. Expose this as a separate research action-model choice. Exact selected-route
   refinement keeps its original single ray. This provider explores multiple
   entries in the same hypothetical opening, so the UI and saved contract must
   state that scope before optimization. Reject unsupported tilted windows,
   fractional transverse origins and sheared source grids; never realign them
   silently. A broader or oblique frontier model needs another explicit version.

Review gates are the seven retained pure-generator tests, existing native
causality/immutable-preview tests, fixed-provider byte-equivalence, state/model
hash invalidation, exact replay, rejected-preview denominator accounting, and
independent source-grid audits of the retained phantom and patient candidates.
Retain the complete-study and early-probe receipts unchanged. A later benchmark
using the new provider requires a new declaration and source snapshot; the
current result alone does not establish an RL advantage or complete planning.
