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
| 1 Cinematic renderer | Closed 2026-11-25 with gates not met (settle 1.95 s, blind crispness 3.25 and depth 3.5 on CTA-A). Kept behind the Volume lighting setting; Shafrir chose to move on. Committed `50f8c119`, pushed. Results in the plan. |
| 2 Segmentation | Done 2026-11-26, unstaged. Gate not met on 018 CT (global b1 9 vs 7); met on 018 MR vs fixed vesselFM. TopCoW-001 confirmation pre-registered, not run. Evidence `phase2-segmentation/README.md`. |
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

## Local storage and instructions (2026-11-25)

- 46 unused local folders were archived to the HDD (`archive/closed-evidence/local-test-outputs-20261125/`,
  `archive/desktop-builds-20261125/`, `archive/local-runtimes-20261125/`); each
  trimmed folder keeps an `ARCHIVED.md` restore pointer. Ledger row added.
- PR #26 (retired docs, tracked `AGENTS.md` index and `CLAUDE.md`) was merged
  into this branch as `154c2b51`. The personal instructions moved to the
  ignored `AGENTS.local.md`, loaded through the ignored `CLAUDE.local.md`.
- The plan is linked from the tracked `AGENTS.md` index (uncommitted until the
  Phase 2 commit, with the ledger edit).

## TopBrain download

`cache/models/topbrain-2025-dockers-zenodo-20158639/` on the HDD. Complete by
size: UZH ct_mr, ARG ct, ARG mr (md5 not yet checked against the record:
UZH 7d04086c…, ARG ct 35d5434f…, ARG mr e1fe74d1…, KDH ct 4b71fe69…, KDH mr
02246627…). KDH ct and KDH mr were downloading at 10 MB/s on 2026-11-25;
resume with `curl -C -` if the files are short of 8.04 GB and 11.86 GB.

## Phase 2

Working decisions (from the coordinator, not to relitigate): no container
runtime; images are unpacked and run natively (MPS or CPU); TopCoW-018 MRA is
the labelled held-out MR reference and PTNT_9937 gets label-free metrics only;
CT models only on CT, MR models only on MR.

- Script: `scripts/topbrain_native.py` (`unpack <image>` streams one archive
  once, checks MD5, writes `inventory.tsv.gz`, `unpack.json` and only app code,
  weights and package metadata to `cache/models/topbrain-2025-native/<image>/layers/`).
- Module: `src/resectionlab/vessel_ensemble.py`, tests `tests/test_vessel_ensemble.py`.
- Evidence and logs: `qc/medivis-parity-20261125/phase2-segmentation/` (logs/).
- HDD is slow for file creation: about 0.9 s per new file while the KDH
  download writes; large reads need a 64 MB buffer or they crawl at 3 MB/s.
- TopBrain v1 labels: CT 1-34 arteries, 35-40 veins (VoG, StS, ICVs, BVRs,
  SSS). MR 1-34 arteries plus 35-42 external carotid branches (ECA, STA, MaxA,
  MMA); MR has no vein classes. All 25 TopBrain training pairs (001-027 minus
  009, 019) are on the HDD, so TopCoW-014/018 are TopBrain training cases.
- UZH image: `opt/app/infer_ct_or_mr.py`, four nnU-Net region models (AM, POS,
  CAR, V for CT; AM, CAR, POS for MR) merged by `label_jsons/label_mapping_*.json`,
  folds (0, 1), no mirroring, Gaussian, step 0.5, components of 12 voxels or
  fewer removed per label.
- vesselFM MRA probabilities for TopCoW-014/018 are being produced with
  `python -m resectionlab.poc_models --model vesselfm-base --modality MRA
  --blend gaussian --output-root <HDD>/cache/poc-model-results`
  (cases `TopCoW-0NN-annotated-mra-benchmark-v1`).
- ARG images (`ARG-ct`, `ARG-mr`, same code, `TRACK` differs): `opt/app/inference.py` ->
  `your_algorithm_v2.py` + `predictor.py`. Stage 1 `resources/weight_1` (binary
  vessel, input z-scored over the whole image); stage 2 `resources/weight_2`
  (49 labels, inputs: min-max image (MR clipped to 0-700 first) + stage-1 vessel
  probability), 6 (CT) / 7 (MR) `checkpoint_best_*` averaged. Their own
  preprocessing: no nnU-Net normalisation, crop to non-zero, trilinear resample.
  Post: keep 6-connected components > 0.05% of the largest; CT maps 43-48 ->
  35-40 and 41-42 -> 0; MR maps 43-48 -> 0. Both datasets list
  `numTraining: 50` = all 25 TopBrain CT+MR pairs, so TopCoW-014/018 are in-sample.
- UZH model folders are named `case_01_27` and their training logs show 2- and
  5-split CV over all TopBrain cases: in-sample too.
- **Memory**: a float32 128x256x256 UZH patch drove the MPS driver to 12 GB and
  pushed 27 GB to swap (internal disk fell to 5 GB free). Now the network runs
  float16 (MPS has no float16 ConvTranspose3d, so those layers stay float32),
  about 4.9 GB and 2.4 s per UZH patch, with
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6` and `LOW=0.5` so it fails instead of swapping.
- Runner used: `PYTHONPATH=src nnUNet_compile=false .tools/poc-model-runtime/venv/bin/python
  scripts/topbrain_native.py predict <team> <case>` (logs under `logs/`).
- First native result (UZH, TopCoW-014 CT, in-sample): 957 s on MPS, peak RSS
  6.6 GB; Dice 0.936, clDice 0.983, b0 error 1 (official labels) versus
  vesselFM Gaussian 0.208 / 0.353 / 341. Scores: `scores/<case>.json`.
- Fixed bug: team label maps were transposed once too often on save (UZH
  014 CT repaired by transposing back; ARG 014 CT started before the fix).
- KDH CT finished downloading at 8,038,457,100 bytes (record size); unpacking.
- KDH image: `opt/app/inference.py` writes the input as float32 NIfTI and
  `run_segmentation.py` runs plain `nnUNetPredictor.predict_from_files` on
  `resources/checkpoint/mulcls` (Dataset200_multiclass, SimpleITKIO, spacing
  0.6 x 0.297 x 0.297, patch 160^3), folds 0-4, `checkpoint_best.pth`, no
  mirroring, step 0.5, no post-processing. It bundles its own nnunetv2 copy in
  `/opt/app/nnunetv2` (not yet diffed against pip nnunetv2 2.5.2).
- Queue helper (scratch, not in repo): runs predictions one at a time and skips
  cases whose `work/<case>/<team>/predict.json` exists.
- KDH CT unpacked, MD5 matches (930 s). Its `mulcls/dataset.json` lists
  `numTraining: 125` with generic names `class1..class40`; no split file ships,
  so assume TopCoW-014/018 in-sample. Its bundled nnunetv2 is a newer upstream
  copy; inference-relevant code (preprocessor, resampling, normalisation,
  sliding window) matches pip 2.5.2 apart from version churn.
- KDH MR archive was truncated (909,830,172 of 11,864,241,848 bytes); the
  coordinator restarted the download and will confirm size and MD5. Do not
  unpack it before that.
- ARG on TopCoW-014 CT: 1,218 s (stage 1 98 s, stage 2 649 s over 6
  checkpoints), peak RSS 8.9 GB.
- Only one native prediction at a time: two GPU jobs at once would swap. The
  queue runs `after.sh <pid> <log> "TEAM CASE" ...` (scratch).
- TopCoW-014 CT (development, in-sample), Dice / clDice / b0 error:
  vesselFM Gaussian 0.208 / 0.353 / 341; UZH 0.936 / 0.983 / 1; ARG 0.931 /
  0.961 / 4; UZH+ARG mean 0.942 / 0.980 / 8; with vesselFM at the default
  rule (0.5, 0.5, gap 1) 0.850 / 0.905 / 8 (16.7k added voxels, mostly wall
  thickening). The rule grid is chosen on 014 CT and 014 MR once KDH is in.
- TopCoW-014 CT, three teams: KDH 0.892 / 0.932 / b0 err 20; UZH+ARG+KDH
  mean 0.936 / 0.973 / 7; plus vesselFM (default rule) 0.843 / 0.900 / 7.
  Annotation itself has b0 9, b1 27. Scores in `scores/TopCoW-014-ct-ensemble-UZH-ARG-KDH.json`.
- KDH MR archive verified by the coordinator (11,864,241,848 bytes, MD5 OK);
  unpacking 20:25.
- KDH MR unpacked (MD5 OK). Its `run_segmentation.py` differs from CT: a
  five-fold binary model (`checkpoint/binary`, Dataset100_binary) multiplies the
  five-fold 42-label model (label x binary). Soft version used here: named
  label = P(binary vessel) x P(label); background takes the rest.
- KDH MR models use `ResEncL` (plans give no architecture arguments; the team's
  `pretrainedTrainer` builds a fixed ResidualEncoderUNet, stages 6, blocks
  1-3-4-6-6-6, features 32-320). The script builds the same; checkpoints load
  strictly. The first KDH MR attempts failed on this and were requeued.
- Sliding window rewritten to evaluate every checkpoint per tile into one
  float16 buffer (nnU-Net sums whole-volume fold outputs, doubling memory;
  KDH MR peaked at 11 GB of CPU arrays that way). Check against nnU-Net on a
  random 2-channel input with 3 ARG checkpoints: identical argmax, largest
  logit difference 1.0 of 81 (float16 accumulation order).
- KDH TopCoW-014 MR took 2,638 s (five ResEncL folds + five binary folds).
- Volumes are now read one channel at a time with 64 MB sequential reads
  (`ChannelArray`); memory-mapped page faults ran at about 10 MB/s on exFAT.
- **Frozen vesselFM rule** (development only, `addition-selection.json`): grid
  low {0.5, 0.7} x high {low, 0.9} x gap {1, 2, 3}; every setting lowered the
  objective on both 014 cases. Least harmful, frozen: low 0.7, high 0.7, gap 1.
  014 CT ensemble 0.936 / 0.973 / b0 err 7 / b1 err 5 -> with vesselFM 0.871 /
  0.911 / 7 / 4; 014 MR ensemble 0.971 / 0.991 / 4 / 1 -> 0.762 / 0.678 / 1 / 15.
- **Held-out CTA-B (TopCoW-018 CT)**, Dice / clDice / b0 err / b1 err
  (annotation b0 9, b1 42): vesselFM Gaussian 0.266 / 0.373 / 357 / 7,
  constant 0.256 / 0.384 / 334 / 14; UZH 0.934 / 0.985 / 1 / 9; ARG 0.891 /
  0.922 / 11 / 18; KDH 0.901 / 0.936 / 36 / 26; ensemble 0.930 / 0.957 / 18 /
  19; ensemble + vesselFM 0.904 / 0.943 / 14 / 20. Dice, clDice and b0 beat
  vesselFM; **b1 does not** (the mean loses cycles: 23 predicted vs 42).
- **Held-out TopCoW-018 MR** (annotation b0 7, b1 31): vesselFM 0.499 / 0.375
  / 1448 / 406; UZH 0.953 / 0.985 / 11 / 6; ARG 0.957 / 0.986 / 5 / 2; KDH
  0.963 / 0.967 / 27 / 10; ensemble 0.969 / 0.985 / 20 / 0; ensemble +
  vesselFM 0.812 / 0.773 / 2 / 51. Gate met on MR for (c) and (d).
- Development scores were redone after freezing the rule (`ensemble --readd`).
- TopCoW 014/018 CT and MR retained under `cases/<case>/<team or
  ensemble-UZH-ARG-KDH>/` (`probabilities.npz` float16 compressed, 1.4-4.6 GB
  -> 2-5 MB each; 119 MB for all four cases); their work files deleted.
- EDEN runs are slow (ARG PTNT_4509 63 min): about 8-9 GB of intermediates
  per team per case on the HDD.
- Coordinator 2026-11-26: the plain mean is attempt one (Betti part of the
  gate missed on 018 CT). **Attempt two**: `select-combiner` chose on 014 CT
  and MR only, same objective vs the mean: weighted (dev clDice) +0.0004,
  maximum -0.062, median -0.016, **base +0.157** (best development team by
  clDice, CT UZH / MR ARG, other teams added where 26-connected at 0.5; module
  `vessel_ensemble.combine`/`vessel_probability`). Frozen in
  `combiner-selection.json`, which also logs every scoring pass of 018.
  Outputs go to `combined-base-UZH-ARG-KDH/`; EDEN uses the same combiner.
- Added `vascular_parity.patch_betti_error` (16 mm cubes over the scored
  region, mean and sum of |b0| and |b1| differences; in `agreement`) and
  `addition_evidence` (vesselFM added voxels split into wall thickening within
  one voxel of the tree vs new branch, with intensity contrast). The merged
  `vessel_with_vesselfm_probability` file is no longer written: vesselFM stays
  its own layer and (d) is scored as tree mask OR added mask.
- Retained volumes are read with `NpzChannels` (one channel decompressed at
  a time) to stay under 3 GB RAM.
- Review fixes 2026-11-26 (all in the script/module/tests): checkpoints keep
  only network weights and the float32 parameter list is freed once networks
  sit on MPS; non-finite logits fail per tile; combiner weights/base stored and
  looked up by team name; held-out passes go to an append-only
  `heldout-log.json` (attempt-one passes back-filled, TopCoW-001 CT/MR
  pre-registered as single-pass confirmation cases, `select-combiner` refuses
  after a held-out pass unless `--reselect`); scores are `scores/<case>.json`
  only (attempt one moved to `scores/attempt-one/`); `ConnectedAddition` has
  no gap (touching only); `base` combiner sums to one; vesselFM run modality is
  checked; `predict` clears stale outputs first; md5-mismatched images are
  refused; `frozen_addition` fails when missing; `--addition` refused on
  held-out cases; root via `RESSECTIONLAB_DATA_ROOT` when set.
- **ARG CT soft path** now sends ARG 35-42 (external carotid branches) to
  background (`ARG_SOFT_MAPPING`); the official map stays as the image makes
  it. ARG 014/018 CT are rerun (queue-9) to keep their raw 48-label stage
  (`stage2.npz`, CT only) for `arg-ct-check`; then selection, combined
  ensembles and scores are redone. The first combined-ensemble run was killed
  before any scoring and deleted.
- Team code licences: only UZH ships one (nnUNet_TopCoW, Apache-2.0); ARG and
  KDH code has no licence file or header (record CC BY 4.0 applies; KDH's
  bundled nnunetv2 copy is upstream Apache-2.0 without its LICENSE file).
- Tests: `tests/test_topbrain_native.py` (loads the script by path); the
  `predict_logits` test needs nnunetv2 and is skipped in `.venv` (passes under
  the model runtime).
- All EDEN predictions finished (KDH PTNT_9937 36 min after the memory fix).
- `arg-ct-check` 014 CT: ARG's raw argmax gives 0 voxels to 35-42 (veins
  43-48: 774-26,174 voxels each), so per-class Dice 35-40 is identical with or
  without them (0.958, 0.949, 0.918, 0.845, 0.548, 0.952); the fix only moves
  soft mass.
- `select-vesselfm` (per modality, on 014): CT threshold 0.3 + largest
  component (Dice 0.337, clDice 0.429, b0/b1 err 8/22), MR 0.9 + largest
  (0.631, 0.501, 6/1). A joint choice kept only the skull on CT (Dice 0).
- `select-combiner` redone after the ARG fix: unchanged, **base** (CT UZH,
  MR ARG), mean objective +0.157 (CT +0.097, MR +0.218).
- vesselFM rule reselected on the combined 014 ensembles (no gap): **low 0.7,
  high 0.9** (every setting still lowers the objective: -0.74 at best).
- Precision check (`precision-check.json`, float16 vs float32 on MPS, real
  centred half-patch tile): largest logit difference 0.05-0.18 (logits up to
  15-168), argmax agreement 0.99999-1.0 for UZH, ARG, KDH CT and KDH ResEncL.
- **018 pass 2** (logged): Dice / clDice / b0 err / b1 err / patch b1 sum.
  CT: vesselFM 0.266/0.373/357/7/50; tuned 0.452/0.530/8/32/24; mean
  0.930/0.957/18/19/12; base 0.910/0.978/4/9/17; base+vesselFM
  0.889/0.965/4/9/18. MR: vesselFM 0.499/0.375/1448/406/146; tuned
  0.651/0.520/6/1/20; mean 0.969/0.985/20/0/10; base 0.952/0.983/7/16/15;
  base+vesselFM 0.811/0.774/2/58/41. Global b1 on 018 CT still misses vs
  fixed vesselFM (9 vs 7); patch b1 meets it. MR gate met.
- EDEN combined ensembles scored label-free (`scores/PTNT_*.json`). PTNT_4509:
  combined-base 215,160 voxels, b0 7, b1 53, 6 floating (3.2%); cleanup layer
  2,476 / 519 / 2,475 (44.9%); vesselFM 2,980 / 520 / 2,979. ARG's MR clip to
  0-700 saturates EDEN TOF (about 20,000): ARG alone 52k voxels, 74% floating.
- Final ensemble per case: `cases/<case>/combined-base-UZH-ARG-KDH/` (TopCoW
  also keeps the attempt-one `ensemble-UZH-ARG-KDH/`). Evidence README and
  `metrics.json` at the phase root. TopCoW-001 confirmation NOT YET RUN.
- Tests: `tests/test_vessel_ensemble.py tests/test_topbrain_native.py
  tests/test_vascular_parity.py` 33 passed, 1 skipped (`predict_logits` test
  needs nnunetv2; passes under the model runtime).

## Phase 3

Stage A (code and tests, no app runs) done 2026-11-25; review fixes 1 to 14
applied 2026-11-26. Unstaged.

- `src/resectionlab/vascular_rooting.py` (`root_vessels`, TopBrain v1 CT/MR
  label schemes) and `tests/test_vascular_rooting.py`. Display wiring in
  `vascular_cleanup.rooted_vessel_layer`, `anatomy_bridge` (scopes `rooted`
  default, `rooted-review`; derived compartments `__vessel_review:`,
  `__vessel_veins:`, `__vessel_arteries:` + layer id), `anatomy_workspace`
  (`vesselDisplayVersion` 3), desktop scope/toggle/colours.
- Rules (frozen on TopCoW-014 and PTNT_4509): gap <= 3 mm, contrast >= 0.5
  or probability >= 0.25, vetoed above 2x contrast, above 700 HU on CT or in a
  bone layer; same-class joins only; LOD from sphere-volume calibre, 1 mm
  always, 0.5 mm within 10 mm of tumour or corridor; unnamed roots = largest
  + inferior-face entries (>= 2 mm, >= 10 mm); lumen/background within 6 mm
  of the closest fragment-tree pair.
- CPU check (fixed 6 mm statistics neighbourhood, 2 mm entry rule):
  floating 6 -> 0 (014), 4 -> 0 (018, held out, run once after the fix; it
  had been run twice before). 014 bridges 3, reviews OAs, an AChA piece and
  L-BVR; 018 bridges 1 (an M3 piece), reviews M2/M3 and OAs. PTNT_4509: 21
  roots, 250 bridged, 51 s, 1.89 GB.
- Stage B (2026-11-26, CPU only; app runs held for "go"): review fixes 1-9
  done. Bridges are face-connected and corner/edge joins get a declared fill;
  the gate counts drawn mesh pieces (`mesh_floating`). Wiring makes the newest
  ensemble the study's initial vessel layers (CTA-A `ensemble-02ff90f79649`,
  CTA-B `ensemble-27af3139bd98`; receipts in `phase3-cleanup/wiring/`, `unwire`
  undoes), all-or-nothing, works on linked reviews. Mesh floating default
  view: 014 annotation 9 -> 0, ensemble 438 -> 0; 018 annotation 11 -> 0,
  ensemble 966 -> 0; PTNT_4509 21 outer pieces, 0 floating. No major vessel
  lost against the references (M2 now major; R-M2 018 92.7 %).
- Wall thickening (2026-11-26): addition pieces with no voxel beyond the tree
  dilated twice go to review (`review.wallThickening`); shown pieces get a
  display-only one-voxel overlap into the tree (`display.overlapVoxels`). The
  mesh gate also joins a vertex lying inside another part's displayed voxels;
  `geometry` prints the mesh count and exits 1 on any mesh floating piece.
  Combined ensemble, mesh floating 0 on CTA-A (3 pieces), CTA-B (6), TOF-A
  (5), TOF-B (3); additions keep 6/3/63/82 pieces, 215/613/6,479/6,755 crusts
  to review. Geometry peak RSS 2.7 to 4.0 GB (3.85 GB on TOF-A without the
  filter). Desktop `verify-vessel-display` counts only "detached components".
- Still to run: re-wire if the combiner changes and for PTNT_4509/9937,
  geometry, hidden captures, sheets, films, `verify-vessel-display` (on a
  CTA-A copy), blind review.

## Phase 4

Stage A (CPU, no app) done 2026-11-26, unstaged. Deviation gate miss recorded
as attempt one; attempt two (non-circular sections) is another agent's and owns
`vessel_tubes.py` and `tests/test_vessel_tubes.py`. Stage B engine and desktop
code done 2026-11-26 without app runs; hidden-app check waits for the GPU.

- `src/resectionlab/vessel_tubes.py` (`centerline-tubes-v1`): `extract`, `mesh`,
  `query`/`capsule_clearance`, `tag_volume`, `rod_sites`, `audit`, `deviation`,
  `to_dict`/`from_dict`. Tests `tests/test_vessel_tubes.py` (12, about 3 s).
  Model and results: `docs/native-vascular-geometry.md` "Centerline tube model".
- Real data (TopCoW 014/018 CT+MR, ensemble and annotation): all walls closed,
  manifold, oriented, zero self-intersections; b0 kept; no named label lost.
  Arterial p95 0.46-0.66 (tube to source) and 0.52-0.85 (source to tube)
  voxels on CTA, about 1.1-1.35 on MRA (0.26-0.41 mm everywhere): gate not met.
- Fixes found on real data: acquired-box clip, balls for empty-skeleton
  components (b0), cross-component shrink, sub-grid smooth-union pockets
  dropped (018 MR b0 27 -> 28 before). Tried and rejected: centroid recentring,
  midrange radii, larger blend.
- Scratch: `/private/tmp/.../scratchpad/phase4/` (keep: `real.py` made the
  real-data table, `table.py` prints it, `result-*.json`); no HDD writes yet.
- Stage B: option `vessel_geometry` / desktop `vesselGeometry` =
  `vessel_tubes.MODEL` sets `operative_vascular_geometry`; model stored in case
  metadata `vessel_tube_models`, digest in the wall receipt; wall, rod sites
  and `vesselTags` derive from it. Files: `surgical_vessel_surface.py`,
  `surgical_patient_wall.py`, `surgical_patient.py`, `surgical_bridge.py`,
  `desktop_bridge.py`; desktop `types.ts`, `case-data.ts`, `contracts.ts`,
  `VolumeRenderer.ts`, `surgicalFieldOptics.ts`, one line in Phase 3's
  `ScanVolumeRenderer.ts`. Tests `tests/test_vessel_tube_engine.py` (6).
- Shell contact (coordinator decision): tube walls only, `surgical_vessel_rods`
  `_project_shell` after each compliant tick (bond-smoothed least squares on
  the wall's vertex transport), bounded by section-motion and sweep checks;
  otherwise the tool stops (`partial`/rejected). Shell overlap at accepted steps
  <= 0.018 mm on press/fast/glancing/axial/junction strokes; press moves the
  vessel 2.2 mm. Tests `tests/test_vessel_tube_engine.py` (8).
- Hidden look (TopCoW-014 CTA crop, scratch `phase4b/look/`): tube wall
  smooth, thin clear-coat highlights; cinematic volume follows the tags with
  no beading (edge speckle until converged). Saved tube cases refuse reopen if
  `vessel_tubes` changes without a new MODEL id.
- Look round 2: tube walls coloured from per-vertex `vesselClasses` (artery
  `#c4553a`, vein `#42668f`); coat roughness 0.3, base 0.24. Operative
  inspection view still lit darker than the volume; highlight broader but
  patchy on v2 wall undulations.
- Default path checked against HEAD code: identical report bytes, case hash,
  snapshots and desktop arrays (scratch `phase4b/compare.py`).
- Attempt two (fit agent, 2026-11-26, unstaged): `centerline-tubes-v2`.
  Per-sample harmonic sections (order 6, RMF frame) plus tip end-cap terms,
  fitted by regularised least squares to owned source wall points; hard-union
  junctions (blend 0); collision via per-segment gradient bound and safe
  radius, `wall_tolerance` = one mesh step; `audit(..., model=)` reports
  `wall_outside_max_mm`. Gate met on all four (arterial p95 0.17-0.39 voxels,
  means 0.06-0.13); audits clean; b0 kept; no label lost. 018 run once per
  modality, MR rerun once for memory (log `scratchpad/phase4-fit/runs-018.log`).
  Rejected: smooth blends with sections (no gain), Newton projection of mesh
  vertices (self-intersections), weaker smoothing (0.3: worse on MRA).
  Scratch `phase4-fit/` (`real.py`, `table.py`, `result-*.json`, `dev.py`).

## Look pass

Cinematic-mode material and bone presentation (2026-11-26, unstaged).
Standard mode untouched.

- Files: `cinematicVolumeShaders.ts`, `CinematicVolumePass.ts` (finish and
  albedo constants, per-sample translucent composite), `volumeTransfer.ts`
  (`cinematicBone: 'shell' | 'solid'`), `ViewerWorkspace.tsx` (Bone select),
  `workspace-view-settings.ts`, `workspace_bundle.py` (+ test); small edits in
  shared `ScanVolumeRenderer.ts` (bone choice, vessel albedo, layers passed to
  the pass). `cinematicTags.ts` untouched (a sigma 5 trial was reverted).
- Final CTA-A (scratch `look/runs/CTA-A/final`): interaction p95 10.2 ms, settle
  1.30 s at 72 samples (Phase 2 CPU jobs running; not a GPU-alone gate run).
- Model: wet vessel lobe + coat (weight 2, roughness .22) + Fresnel studio reflection; key disc cos .985; vermilion artery
  `#b64f37`, vein `#4a70a8`; porcelain bone `#e6d9d5` (20 % layer tint) with
  density-tinted albedo and warm wrap; see-through shell keeps 30 % of its
  diffuse light; two-voxel lumen normal stencil at rest; back-facing
  silhouette normals projected edge on; shadow/bounce origins stepped back
  along the eye ray; bounce from nearby matter includes half an unshadowed key.
- Dark speckled fringe (coordinator report): translucent layers were
  composited over the running mean with one sample's depth; now composited
  per sample before accumulation. Dark dotted rims on CTA-A came from
  mirrored silhouette normals; fixed by projection.
- Bone default: unset = solid when cut or skull opaque, shell otherwise
  (solid uncut CTA-A hides every intracranial vessel and the tumour).
- Iteration captures: scratch `look/runs/CTA-A/<it>` via `look/look_capture.py`.
- Found and fixed: `workspace_bundle._viewer_state` rejected
  `volumeRendering.mode`, so a cinematic native view could not be saved.

- Review round (2026-11-26): edits cut the stored model (ordered `removedCells`
  + open faces, `with_removed`/`limits`; far vertices bit-identical, no refit);
  shell bookkeeping (charged hand, projected-pose contact, projection kept out
  of the carried velocity); option refused without V2 physics and working
  axis; per-wall shell contact; tube metadata tied to its policy; `query`
  closest point by Newton steps, NaN beyond reach; coplanar overlaps in
  `audit`; `deviation(source_classes=)` with unowned counts; display-only
  normal smoothing. Final `combined-base-UZH-ARG-KDH` outputs are not on the
  drive; re-measured on `ensemble-UZH-ARG-KDH` (`phase4-fit/real2.py`).
