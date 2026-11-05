# Integrated acquired-anatomy engineering suite

This extends the immutable `2026-10-31-data-foundation-v1` release. Bulk inputs,
review evidence, recipes and packages are under `integrated-v1/` on the configured
external data root. The resumable ledger is `integrated-v1/ledger.json`.
The current accepted catalog and its checksummed independent decisions determine
capabilities; a native preparation flag does not grant acceptance.

## Reusable acquired CTA anatomy hosts

TopCoW 004, 006, 012, 014 and 018 each have a `CTA-ANATOMY-v1` anatomy base.
Each contains the original CTA in its native RAS+ grid, a coarse
brain estimate, a local dense skull estimate derived from that CTA, and two
partial vascular layers from the supplied artery and vein/sinus annotations.
Each package retains six acquired originals and its existing TRAIN family.

These baselines have no modeled target. The case reports bind the paired
synthetic variants and prove that they share the same image, frame and four
anatomy arrays. The target sits in the retained brain membership, so the
baseline contains brain at those cells. Adding a tumor remains a separate
scenario decision. The CTA stays the acquired reference in both packages.

The brain estimate may include CSF. The local skull mask covers part of the
calvarium. Skull base and cropped anatomy remain uncertain. The vascular masks
cover selected vessels and sinuses. The host packages support anatomy inspection
and scenario preparation. Instrument permissions and histories belong to the
separate reviewed task cases.

On the external data root, `integrated-v1/anatomy-hosts-v1/candidate-index.json`
binds the native inputs, generated packages and preparation reports. To reproduce
the anatomy preparation in a new output version, use the existing environment:

```sh
.venv/bin/python scripts/data_integrated_hosts.py --subject 006 --version reproduction-v1
```

## Original integrated CTA rehearsal variants

TopCoW 004, 006, 012, 014 and 018 from the acquired TopBrain v3 (2026-08-17)
release retain their original TRAIN assignments and
clinical-cohort status. Each integrated case preserves acquired CTA intensities,
physical geometry, supplied arterial/sinus arrays and embedded originals. A
source-derived local intracranial envelope, a dense skull estimate and a
geometry-only synthetic target share that exact native frame. No borrowed anatomy,
array resizing, cross-modality affine copying or synthetic CT signal is used.

The brain representation is a **partial superior intracranial envelope**, including
CSF uncertainty. Its flat inferior boundary is artificial. The dense CT skull
estimate is supported by original image intensities and independently inspected
source context; it is neither a complete skull nor segmentation ground truth.
Calcification and partial-volume uncertainty remain. Arterial and superior
sagittal sinus annotations are incomplete. Missing labels and crop faces are
unknown. Source CTA/MRA subject pairing does not establish acquisition timing or
co-registration; these CTA cases do not fuse the separate native MRA.

Intact-baseline proposals intersected source scalp and were rejected. The
separate conditional task explicitly assumes prior superficial exposure and
unmodeled dural opening. Neither event is observed or simulated. Acquired CTA
remains a **baseline reference before assumed exposure**. This is not an intact
head access sequence or evidence of clinical safety.

Every admitted instrument action is bounded by an independently reviewed native
voxel-face task box. The production bridge checks whole tool capsules, both hands,
rotation, movement and withdrawal. It restricts unknown-space actions atomically
and refuses autonomous planning/preparation commands that do not implement this
contract. It preserves catalog tool dimensions and existing material guards.
The box is a task restriction, not an anatomical label or a measured free space.

## What the physical workflows establish

The actual public production bridge executes fixed drill/CUSA sequences, commits
real material removal, withdraws both instruments and saves a replayable native
project. Full tool/FOV admission, clock, removals and thermal history are retained.
Saved histories are replayed onto metadata-only guarded v3 sources through public
commands, then verified by a cold reopen/replay. Native and application evidence
are independently reviewed before promotion.

All five finite acquired-case protocols encounter the retained skull/brain
boundary and stop before reaching the synthetic target. They support conditional
local drilling and **negative access/material-boundary regression**, not successful
target resection. The source-bound finite offset continuation for 006 also stops;
its previous drilling and thermal burden are preserved. No obstacle was erased,
tool reduced or constraint weakened. Failure of these protocols does not prove
that every other approach is impossible or that a patient is clinically ineligible.

Modeled temperature is uncalibrated; historical peaks are approximately 124.3 °C.
A final spatial temperature maximum is not the historical maximum. These runs
must not be described as thermally safe. The separate generated control establishes
public drill/CUSA/target/withdraw/replay behavior, adds zero source people and does
not demonstrate acquired-case target reach.

Primary target corridors do not intersect supplied vascular labels, so they do
not by themselves establish vascular causality. Any separately reviewed sinus
conflict test has its own scope, source contacts and public response evidence.
Do not infer complete vascular avoidance or clinical clearance from a clear path.

## Healthy MRI/MRA subset

IXI077 and IXI081 retain their existing TRAIN families. Original T1 is unchanged;
original native MRA is preserved separately and remains unavailable for fusion.
An explicit six-degree-of-freedom rigid registration resamples acquired MRA and
public model-assisted/refined vessel annotations onto the T1 grid. Images use
linear interpolation; labels and observed-MRA coverage use nearest neighbor with
the same physical pull transform. Synthetic 4 mm radius targets remain geometry
only. The MRA slab's missing coverage remains unknown; neither person has skull CT.

Independent numerical and three-plane source/overlay inspection permits
**registered overlay inspection with alignment uncertainty**. No independent
patient landmark error bound was established. Precise navigation, clinical vessel
clearance, measured-skull drilling and complete surgical rehearsal remain
unsupported. The application displays these limits and uses identity mapping for
the already resampled image, preventing a second application of the original rigid
transform. Original rigid transforms, annotation/source hashes and review receipts
remain in provenance.

## Open and reproduce

```sh
export RESSECTIONLAB_DATA_ROOT="/Volumes/shafrir/old coding projects/RessectionLabData"
./scripts/launch_poc.sh
```

Use **Cases** for the currently accepted catalog. The external
`integrated-v1/START-HERE.md` identifies exact canonical cases and physical replay
files. **Data** permits the single reviewed derived MRA blend; original native MRA
stays separate. CTA **Rehearse** exposes the conditional-stage warning and task
restrictions. A saved physical history opens through **Verify saved replay**.

Inspect pinned recipes before regenerating; existing directories are immutable.
Use a fresh version token rather than overwrite evidence:

```sh
# Preparation-only dependencies; existing core NumPy/SciPy/nibabel stay required.
.venv/bin/python -m pip install -e '.[data-preparation]'
.venv/bin/python scripts/data_integrated_anatomy.py --root "$RESSECTIONLAB_DATA_ROOT" --subject 006 --version enclosure-reproduction-1
.venv/bin/python scripts/data_integrated_cases.py --help
.venv/bin/python scripts/data_integrated_healthy.py --help
.venv/bin/python scripts/data_integrated_probe.py --help
.venv/bin/python scripts/data_integrated_rebind.py --help
```

Preparation can reuse the existing checked derivatives. The anatomy command above
creates a fresh immutable version and refuses existing output. Per-case reports bind exact source,
anatomy, target, reviewed domain and generator hashes. Probe contracts enumerate
finite actions and bind the original case; preserve them as evidence instead of
retuning targets around outcomes. Failed optional segmentation model experiments,
intact-baseline candidates and superseded UI provenance captures are retained on
the drive with dispositions, excluded from new accepted discovery.

For example, reproduce the frozen 006 first-pass protocol with a fresh output name
(it is expected to stop before the target):

```sh
.venv/bin/python scripts/data_integrated_probe.py \
  --case "$RESSECTIONLAB_DATA_ROOT/integrated-v1/cases/TopCoW-006-CTA-INTEGRATED-integration-conditional-stage-v2/TopCoW-006-CTA-INTEGRATED-integration-conditional-stage-v2.ressectionlab" \
  --contract "$RESSECTIONLAB_DATA_ROOT/integrated-v1/probes/006-conditional-route-contract-v3/contract-reviewed.json" \
  --name 006-first-pass-reproduction-1

# Reproduce the complete already proven history under the persisted v3 guard.
# This adds zero actions and compares physical clock, temperature and removals.
.venv/bin/python scripts/data_integrated_rebind.py \
  --history "$RESSECTIONLAB_DATA_ROOT/integrated-v1/probes/006-offset-continuation-execution-v1/replay.ressectionlab" \
  --source "$RESSECTIONLAB_DATA_ROOT/integrated-v1/cases/TopCoW-006-CTA-INTEGRATED-integration-conditional-stage-v3/TopCoW-006-CTA-INTEGRATED-integration-conditional-stage-v3.ressectionlab" \
  --name 006-guarded-reproduction-1 \
  --evidence "$RESSECTIONLAB_DATA_ROOT/integrated-v1/probes/006-offset-continuation-execution-v1/report.json"
```

Healthy registration may be reproduced from the saved source receipt, method and
transform files in `healthy/IXI077/rigid-v1/` and `healthy/IXI081/rigid-v1/`.
The helper refuses overwriting completed versions. Do not substitute the original
MRA-to-T1 forward matrix as the pull transform or apply it again to the already
registered auxiliary image.

Local publication can append exact-byte independently accepted additions while
preserving every row and receipt from the prior frozen release:

```sh
.venv/bin/python scripts/data_rebuild_release.py \
  --decisions integrated-v1/qc/independent/release-decisions-v1.json \
  --inherit catalog/releases/2026-10-31-data-foundation-v1.json \
  --freeze NEW_IMMUTABLE_RELEASE_NAME
```

Only run after independent case/application decisions are complete. This performs
local catalog publication, not network upload. The new manifest preserves the
base release identity and records the new review authority separately. Do not
rerun the legacy publisher without these arguments to replace an integrated
catalog with the older decision set.

## Verification scope

Focused checks cover physical target construction and overlap, source/numeric
immutability, strict derived registration and no double mapping, native persistence,
full capsule task-domain enforcement, generated public workflow controls, and
append-only release authority. Node tests cover actual image hydration and acquired
versus generated image provenance. The frontend is type-checked and built. Actual
Electron checks run hidden, with source slices, rendering, save/reopen and relevant
fusion/rehearsal behavior captured on the external drive.

An existing bridge undo/reopen frame-token assertion fails identically with the
HEAD bridge implementation; the differential is retained under
`integrated-v1/qc/task-domain-validation/`. This is not an all-repository test pass.
The broader research work in this shared checkout is separate. No RL training,
medical-image upload, account creation, clinical validation or push is part of this
integration cycle.
