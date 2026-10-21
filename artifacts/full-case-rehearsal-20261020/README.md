# Full case rehearsal: collaborator handoff

This branch includes the saved sub-06 rehearsal, its intact source workspace,
protection arrays, meaningful intermediate checkpoints, original action history,
canonical traces, review scripts, test receipts, and actual Electron screenshots.
These specifically selected files under `testcases/` are tracked despite the
normal local-data ignore rule. No additional patient download or model inference
is needed to open this saved case. Model weights are not needed or included.

The numerical/application work was frozen at
`4087a8fa48f68712cb6acf8883a9ea38d4dbda50`; subsequent handoff commits add the
portable entry points, documentation and data. Original evidence retains its
historical commit, timestamps, host paths and hashes. Do not interpret those
historical paths as paths you must recreate on your computer.

## Clone and set up a Mac

```sh
git clone --single-branch --branch codex/poc-first-rehearsal https://github.com/shafsthegoat/RessectionLab.git
cd RessectionLab
```

The payload is large. Ordinary Git contains the files; Git LFS and a separate
release download are not required. Do not use a blob-filtered clone if you need
an offline copy of all evidence.

Install Python 3.12, Node.js 22.17 or newer, and pnpm 11. The original environment
used Python 3.12.12 and pnpm 11.25.0 on Apple Silicon macOS. Then run:

```sh
./scripts/setup_shared_rehearsal.sh
.venv/bin/python scripts/verify_shared_rehearsal.py
```

Setup explicitly installs the existing pinned Python requirements and desktop
lockfile. It downloads dependencies, not models or additional datasets, and does
not replay a patient. The dependency set includes PyTorch and is substantial.
Dependency resolution with `--dry-run --ignore-installed` was checked on the
authoring Apple Silicon Mac. This handoff did not perform a new clean dependency
installation or qualify an Intel Mac or older macOS release; strict replay
identity checks must remain enabled on other runtimes.

The default verification checks published file sizes/SHA-256, original arrays
and source references, endpoint state hashes, protection cells, branch prefixes,
target accounting, fluid bookkeeping, and withdrawn tool/hold state. It executes
**zero patient physics actions**. For a standard-library-only byte check before
installing dependencies, use:

```sh
python3 scripts/verify_shared_rehearsal.py --files-only
```

## Open the actual saved rehearsal

With a secondary monitor connected:

```sh
./scripts/open_shared_rehearsal.sh
```

On your own single-monitor Mac, explicitly allow the ordinary display:

```sh
./scripts/open_shared_rehearsal.sh --allow-primary-display
```

This builds and opens the existing Electron application. The wrapper locates
the repository from its own path, so the clone can live anywhere, including a
directory containing spaces. Do not use the historical machine-specific
`final/open_rehearsal.sh` inside the evidence tree.

The native file is:
`testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab`

Its SHA-256 is
`095659e54f381355e8f5655b6c5907990b2ac4767dc9ce58297e3bc4d6f3268d`.

In **Rehearse**, choose `full-primary` (2,148 actions) or `full-vascular`
(73 actions). Rewind the timeline and press **Play replay**. Pause, previous/next
state and the timeline slider expose exact recorded boundaries. Observed
microsteps are read only; use **Return to recorded state** before editing or
branching. Use **Branch**, then **Save** to a new filename when testing changes.
Keep the published native and intact source unchanged. **Fit 3D** frames current
tool bodies; **Frame operative field** provides a closer display-only view.

**Opening Rehearse on a new machine can be very slow.** Its private verified
cache starts empty and the app must replay the recorded physical actions before
trusting the saved endpoint. The original forced primary verification took
about 4 h 16 min; this is historical timing, not a prediction for your Mac.
Subsequent opens can reuse that computer's authenticated cache. No private
cache keys or trusted caches from the original machine are distributed, and no
verification guard is bypassed. Both cache budgets are explicitly bounded to
8 GiB by the wrapper; leave additional room for new outputs and dependencies.

**Playback itself is also very slow:** early actual Electron playback advanced
63.8 simulated seconds in 631.1 real seconds despite selecting 4×. Rendering is
a plausible bottleneck, but the cause was not conclusively isolated. The user
deferred performance fixes. Do not interpret the speed selector as achieved
throughput. You can inspect the recorded screenshots and results immediately
without waiting for a full replay.

## Re-run the physics verification explicitly

These commands use the same public engine and existing independent audit helper.
They do not launch an autonomous director, add resection actions, infer anatomy,
or train a policy. Choose one branch at a time:

```sh
.venv/bin/python scripts/verify_shared_rehearsal.py --canonical vascular
.venv/bin/python scripts/verify_shared_rehearsal.py --canonical primary
```

Each command prints its unique output directory under ignored
`testcases/outputs/shared-rehearsal-verification/`. The canonical audit disables
cache reads, checks every action, source identity, protection and conservation,
and publishes only the successfully verified local cache. On a different
numerical build, a state-identity mismatch is a failure to investigate, not a
reason to accept the saved state without replay.

After a successful canonical run **on your machine**, prepare actual observed
frames and descriptive 0.2-second telemetry in one further public replay:

```sh
.venv/bin/python scripts/verify_shared_rehearsal.py --prepare-frames vascular
# Optional long primary preparation:
.venv/bin/python scripts/verify_shared_rehearsal.py --prepare-frames primary
```

Once preparation finishes, audit every cached frame without physics fallback.
Supply the canonical report from your own preceding run, not the historical
report from the authoring computer:

```sh
.venv/bin/python scripts/verify_shared_rehearsal.py --audit-frames vascular --canonical-report /absolute/path/to/your/vascular-canonical.json
```

The same command works for `primary` with its own local report. The frame audit
uses Python 3.12 `sys.monitoring`. Do not run these helpers with `python -O`.

For targeted generated engineering and desktop regressions:

```sh
.venv/bin/python -m pytest tests/test_surgical_tools.py tests/test_surgical_vessels_v2.py tests/test_full_case_cleanup.py tests/test_full_case_rehearsal.py tests/test_surgical_frame_replay.py tests/test_surgical_verified_cache.py --basetemp="testcases/outputs/friend-tests-$(date +%Y%m%dT%H%M%S)" -q
pnpm --dir desktop test
```

These tests include synthetic fixtures; they are not additional patient evidence.
Historical live-console recovery loaders and `final/verify_and_prepare.py` are
preserved for inspection, **not** portable run-all entry points. They pin old
live-session boundaries, output destinations or the original commit. Use the
commands above instead.

## What was achieved, and what remains unverified

| Measure | Primary | Vascular alternative |
| --- | ---: | ---: |
| Original owned target | 43,573 mm³ | same source |
| Owned target removed | 35,444 mm³ (81.34%) | 312 mm³ |
| Owned target retained | 8,129 mm³, 161 six-connected regions | 43,261 mm³ |
| Bone removed | 16,552 mm³ | 3,453 mm³ |
| Actions / simulated seconds | 2,148 / 4,500.852 | 73 / 202.569 |
| Active wounds at endpoint | 0 | 0; one persistent sealed wound, five nonpatent faces |

Both branches retain their genuine 42-action source-bound common prefix.
The full raw tumor annotation is 45,115 mm³; 35,810 mm³ of that annotation was
removed in the primary. Owned material precedence gives the separate 43,573 mm³
task denominator. Zero owned non-target soft-tissue removal does **not** establish
zero clinical injury: removed cells include 36,402 raw brain-annotation
memberships, and the overlapping estimates are unreviewed.

The primary stopped as a **budget-limited engineering rehearsal with residual
target**. It is not complete clearance, maximal feasible removal, or proof that
residual target is unresectable. Real recorded collision/material constraints
are distinct from controller search limits and untested routes (superior cursor
26/99). Bounded field cleanup retained 15,280.03 mm³ of debris after 1,271.97 mm³
aspiration. Both instruments were withdrawn/inactive and all holds released.

Canonical source/history/conservation checks passed for both branches. All
2,148 complete primary outcomes, including contact positions, matched the
preserved original live history. The vascular branch actually recorded injury,
passive bleeding, bipolar sealing, irrigation and suction. It generated
4.40016 mm³ modeled blood plus 6 mm³ irrigation; 5.96624 mm³ was suctioned,
4.32421 mm³ retained and 0.109705 mm³ escaped. Bone/debris and fluid conservation
passed; primary thermal residual 8.45e−7 J was within the declared 1e−6 J tolerance.

Actual GUI checks covered early playback, pause, rewind/return to a boundary,
branching, Save to a validation copy, fresh-process reopen and both endpoints.
**Full final primary/causal-vascular GUI playthroughs and final 0.2-second
telemetry/all-frame coverage remained unverified when the user approved closeout.**
Older vascular playback/telemetry receipts are explicitly historical versions.
The authoring machine's in-progress final frame cache and active telemetry stdout
log are not included or claimed complete. No video was requested at final closeout.

Missing dura/scalp/attachment/functional evidence and vascular completeness
remain UNKNOWN. Tool/material, deformation, thermal, wound and flow laws are
uncalibrated research assumptions. Six tool families have actual patient-action
evidence; the other nine catalog families retain labeled synthetic-fixture
checks. This is an engineering rehearsal, not a clinical treatment result.

## Evidence and data provenance

* [Machine-readable payload inventory](manifest.json): every published payload
  file and the unchanged engine/desktop sources, with sizes and SHA-256 hashes.
* [Closeout receipt](../../testcases/outputs/full-case-rehearsal-20261020/final/closeout-receipt.json)
  and [independent completed-canonical review](../../testcases/outputs/full-case-rehearsal-20261020/final/independent-completed-canonical-review-20261021.json).
* [Original live history](../../testcases/outputs/full-case-rehearsal-20261020/final/live-primary-history.jsonl),
  [primary canonical trace](../../testcases/outputs/full-case-rehearsal-20261020/final/primary-canonical.actions.jsonl)
  and [tool-family inventory](../../testcases/outputs/full-case-rehearsal-20261020/final/tool-family-inventory-source-records.json).
* [Fresh primary endpoint](../../testcases/outputs/full-case-rehearsal-20261020/final/gui/fresh-primary-endpoint-0009.png),
  [fresh vascular endpoint](../../testcases/outputs/full-case-rehearsal-20261020/final/gui/fresh-vascular-endpoint-0010.png)
  and [GUI observation index](../../testcases/outputs/full-case-rehearsal-20261020/final/gui/observations.json).
* [Source preparation/provenance](../../testcases/environments/meningioma-sub-06/report.json)
  and [source protection manifest](../../testcases/environments/meningioma-sub-06/rl-protection.json).

The source record identifies the acquired imaging as sub-06 of
[A Paired Head CT-MRI Dataset for Cross-Modality Image Synthesis, Zenodo 17486320](https://zenodo.org/records/17486320),
under CC BY-NC 4.0. Supplied masks are model-derived estimates, not reviewed ground
truth. Original provenance retains the separate model notices, including the
SynthStrip/FreeSurfer notices and vesselFM Open RAIL++-M research/noncommercial
restrictions. Preserve the source attribution and notices when sharing derivatives.
No pretrained weights or inference runtimes are redistributed in this payload.

Excluded runtime material is listed in the inventory: disposable pytest trees
(including private test keys and sparse oversized-file fixtures), Python bytecode,
executable pickle snapshots, host-authenticated caches/private keys, redundant
fixture-only saves, invalid V8 profiling captures and recorder-test media. Their
meaningful test/review reports and relevant source remain included. No existing
live engine was killed or restarted to publish this handoff.
