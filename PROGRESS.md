# Branch progress: shaf/medivis-vascular-parity-20261125

Temporary working log for this branch only. Agents: read this first instead of
re-deriving state, and update it when a step finishes or a fact changes.
**Delete this file in the last commit before the pull request is opened.**
The goal, gates and metric definitions are in
[docs/MEDIVIS-PARITY-PLAN.md](docs/MEDIVIS-PARITY-PLAN.md); do not repeat them here.

## Rules that are easy to forget

- Commits: author and committer `shafsthegoat`, no trailers, one commit per
  feature, push after each phase, never merge. Check `git log` before pushing.
- Subagents: Claude Opus 5.5, high effort. A fresh read-only reviewer per review.
- No worktrees. `stash@{0}` holds earlier unrelated work: never touch it.
- All app runs hidden (`--automation-hidden` harness). Evidence, scratch and
  downloads go on the HDD under `qc/medivis-parity-20261125/`; verify
  `.ressectionlab-data-root.json` first. EDEN data never leaves the Mac or HDD.
- `pnpm` is absent. From `desktop/`: `node_modules/.bin/tsc --noEmit`,
  `node_modules/.bin/vite build`, `node --test electron/*.test.cjs`,
  `node --experimental-strip-types --test tests/*.test.mjs`, and the same for
  `src/viewer/*.test.mjs`.
- Measure on the built-in Retina panel: pass `--display-scale 2` to the harness
  (`scripts/capture_vascular_parity.py` does). An external 100 Hz 1x monitor is
  sometimes the main display and silently changes pixel ratio and refresh rate.
- The HDD hung once under a large download running beside captures. Do not run
  bulk writes to it during captures; probe it with a timeout if a command stalls.

## State

| Phase | Status |
| --- | --- |
| 0 Baseline | Done, committed `74f8e23b`, pushed. |
| 1 Cinematic renderer | Closed 2026-11-25 with gates not met (settle 1.95 s, blind crispness 3.25 and depth 3.5 on CTA-A). Kept behind the Volume lighting setting; Shafrir chose to move on. Results in the plan. |
| 2 Segmentation | Not started. TopBrain archives partly downloaded (see below). |
| 3 to 5 | Not started. |

## Phase 1 notes worth keeping

- Evidence index: `phase1-cinematic/README.md` on the HDD. Results table: plan.
- `report.views[].pane` is already in backing pixels; never multiply it by
  `captureScale` (that mistake produced a wrong "48 samples pass" reading).
- Settle is 150 ms hold plus 72 samples at about 25 ms; 1.5 s needs 18 ms.
- Thin vessels are beaded because opacity follows the binary label mask; soft
  probabilities (Phase 2) and implicit tubes (Phase 4) are the intended fix.
- TOF blind scores did not move: fragments dominate (Phase 3).
- The later Phase 1 edits (fence window, lumen-gradient normals, display
  pinning, test hardening) were made after the independent review.

## Open decisions for Shafrir

1. TOF held-out scoring: no EDEN annotation exists. Proposal: score labelled
   TopCoW-018 MRA, judge PTNT_9937 on label-free metrics and blind review.
2. TopBrain dockers are linux/amd64 with NVIDIA GPU. Plan: unpack the archives
   and run the weights natively; not yet confirmed possible.

## TopBrain download

`cache/models/topbrain-2025-dockers-zenodo-20158639/` on the HDD. Complete:
UZH ct_mr, ARG ct (md5 not yet checked). Partial: ARG mr. Missing: KDH ct, KDH mr.
Resume with `curl -C -` when no capture is running.
