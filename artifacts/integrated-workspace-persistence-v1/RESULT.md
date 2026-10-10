# Integrated workspace persistence

The Mac app now saves and reopens the primary case, attached native-grid images,
source/estimated label provenance, each image's cursor and layer visibility,
selected image, and the backend-owned generated episode with its selected replay
frame. Native Save and fresh-process reopen were exercised on this Mac. This is
software integration evidence; it is not anatomical or physical validation.

## Working behavior

The optional extension lives inside the existing `.ressectionlab` case ZIP.
Primary semantic/planning identities stay unchanged. Auxiliary images include
actual immutable source arrays, so originals need not remain at their previous
paths. Their separate native frames, unknown association/registration/time and
display-only status survive. They remain outside the planning case registry and
are absent from policy input channels. Old plain case bundles remain readable
and clear stale same-identity auxiliary/replay state.

The backend owns saved episode bodies. Reopen re-executes the known fixed
generated task and checks its complete non-timing history, geometry and accounting;
a recomputed digest alone cannot promote an edited replay. Two named historical
timing fields are bounded and explicitly not remeasured. Archive membership,
duplicates, array sizes/dtypes, source hashes, frames and view/replay associations
are checked before transactional installation. A bad reopen preserves the current
workspace. No Python pickle is used.

Desktop hydration validates and loads the complete source session before
publishing it, including generation checks when reopening the same case hash.
Selecting an auxiliary image hides primary planning/replay. Open and Import are
available on that image when idle; busy, read-only and stopped-engine guards stay.

## Executed checks

- Canonical backend/imaging/bridge selection: 54 passed, one skipped in 23.45 s.
  The skipped cross-language path passed explicitly with the Node executable
  supplied (one pass, 0.33 s): 55 relevant checks overall. An intervening mistyped
  selector selected zero tests and its exit-5 receipt is retained.
- Worktree desktop: 310 passed (83 main, 158 renderer, 69 viewer), zero failures.
- Independent candidate review: 18 backend and 28 desktop controls, plus actual
  generated backend save → fresh reopen → asset registry → 96-frame hydration.
  Its receipt predates the final navigation correction and root live checks.
- Worktree production build passed before/after navigation correction. The exact
  staged desktop, excluding unrelated neighboring-inspection edits, also builds.
  Vite's existing >500 kB chunk advisory remains; packaging was not repeated.
- Navigation correction: both App projections typechecked and 32 availability
  combinations each passed; root observed idle/busy behavior on actual FLAIR.

See [source bindings](source-index.json), [independent review](independent-review.md)
and the unaltered compact logs in this directory.

## Actual Mac observations

[The live receipt](live-workspace-check.json) binds the saved bundles and owned
process measurements. The generated auxiliary image reopened at RAS 103/6/10 mm;
primary cursor 6/6/4 mm and hidden target layer remained separate. Switching to
primary and showing Episode restored frame 3/95, tool tip 6/6/2 mm, zero target
removal and 1 mm³ other removal. Saving with replay visible reopened directly to
that same Episode frame and state hash. Screenshots were inspected through
computer use; no screenshot file or image arrays are committed.

Public RESECT Case4 T1 and FLAIR were saved through native dialogs into a
29,960,302-byte local bundle. A fresh app reopened the selected FLAIR at
RAS 6.5/6.8/−33.3 mm, with both 256×256×192 sources retained. The primary T1 case
and planning hashes are unchanged. The scans have different source-reported
frames; registration, anatomical orientation and planning suitability remain
unreviewed. The pre-existing orientation question documented in the preceding
integration record remains open. No source affine was changed to improve visuals.

Five owned UI sessions exited normally; the largest sampled process-tree RSS
was 1,132,740,608 bytes, below the 1.5 GiB guard. GPU memory and sub-sample peaks
were not measured. The original acquisition worker was not interrupted.

RESECT attribution: Xiao, Fortin, Unsgård, Rivaz and Reinertsen (2017),
CC BY 4.0, DOI 10.11582/2017.00004; [source release](https://archive.sigma2.no/dataset/5D6BFC33-F58D-4F56-88E8-C40AF269D6F2).
This use is local display/persistence checking, not training or held-out planning.

## Reproduce and remaining scope

Run `.venv/bin/python -B -m pytest -q tests/test_workspace_persistence.py`
and the desktop test/build scripts. In the app, open a case, attach a supported
scalar NIfTI (with optional aligned source or estimated labels), select it, Save,
quit and reopen the saved workspace. On a generated Episode, select a recorded
frame, Save, quit and reopen to inspect the same replay.

Camera pose, layout, contrast, global opacity and importer defaults are not saved
by this extension; cursors, per-layer visibility and replay selection are. Only
the current fixed generated episode contract can be restored as authoritative
simulation replay. Auxiliary scans are not registered/fused, segmentations are
not automatically qualified, and mode-aware learned behavior and physical/real-
patient generalization remain open. The next integration joins existing vascular
encounter accounting to the same sealed episode and desktop replay without
exposing private labels to the planner. No new optimizer update or RL performance
claim occurred in this persistence slice.
