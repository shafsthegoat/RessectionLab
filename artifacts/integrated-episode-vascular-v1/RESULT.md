# Vessel encounters linked to the shared episode

The generated desktop episode now has a post-execution vessel-annotation evaluator
using the same complete shaft/tip insertion and reverse-withdrawal history shown
in replay. The new **Evaluate annotated encounters** action works for scripted
and SEARCH episodes, including a reopened saved episode. It adds a separate scalar
report; actor inputs, rewards, transitions, tissue state and workspace persistence
are unchanged. It cannot accept a caller-supplied patient, reference mask or path.

The frozen generated reference is loaded only after the episode, native actions,
replay states and independent full-tool geometry checks pass. The result is bound
to source, strategy, episode and physical-history identities through Python, host
IPC and renderer validation. Callback mutation and replaced cached episodes refuse
publication. The independent review's original geometry-alias defect is retained
under `negatives/`: it once scored changed geometry while retaining the original
history identity. Detached snapshots and repeated binding checks repair it.

## Executed checks

- Canonical backend/bridge/persistence/native-episode selection: **69 passed** in
  63.59 s. The initial mistyped test filename collected no tests (exit4), preserved.
- Desktop: **333 passed** (87 host,177 renderer,69 viewer), no failures. Worktree
  and exact staged clean desktop both typecheck/build. The existing >500kB chunk
  advisory remains. Author's earlier transient Vite cleanup failure and successful
  isolated rerun remain preserved; root's complete run was clean.
- The canonical backend report passes actual host and renderer validators, and a
  separately retained direct evaluation exactly equals the bridge result.
- Independent source audit matches all promoted reviewed source pins; canonical
  test changes remove only ignored-stage imports and adjust the output root.
- Root clicked through the actual Mac application: reopened scripted evaluation,
  aspiration/probe/STOP/initial states, incomplete coverage, new SEARCH execution,
  prior-report invalidation and new SEARCH evaluation. Screenshots were inspected.
  Normal quit:92.80 s session,724,140,032 B sampled process-tree peak, exit0/reaped.
  Transient peaks and GPU memory were not measured. No new packaged app was built.

The retained scripted report contains six actions, one unique positive reference
cell and one unknown in-grid cell over the full tool path (91 touched cells).
Shaft and tip counts overlap; their sum is not the union. Displayed counts concern
the **whole recorded action and return path**, not instantaneous frame contact.
STOP has no sweep. Reference coverage is annotation coverage, not complete vessel
knowledge. Contact is separate from removed overlap, which remains unassessed.
Clinical injury and biological clearance remain null; patient admission is false.

## Reproduction and remaining scope

Use the two canonical vascular test files, related shared-episode/persistence
checks, and `pnpm test`/`pnpm build` in desktop. `reproduce-canonical.py` records a
new generated bridge result and durable direct evaluation beside its own location;
copy it into a fresh ignored directory before running, then pass its JSON fixture
to `desktop/tests/episode-vascular-crossstack.mjs`. Existing receipt directories
are exclusive and must not be overwritten. The committed evaluation is compact
metadata; generated arrays, app bundles, patients and weights are not included.

This establishes shared software accounting, not patient anatomy, vessel injury,
force accuracy or a learning benefit. Evaluation remains ephemeral in the UI.
Real-case vessel evidence, uncertainty-aware decisions, physically validated
interactions and matched learned-policy transfer remain separate unfinished work.
