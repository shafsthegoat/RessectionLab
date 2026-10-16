# IXI265 tissue estimates work in the native anatomy viewer

The separate tissue workspace opens in actual Electron with CSF-like,
gray-matter-like, white-matter-like and eight-voxel UNKNOWN layers. Individual
visibility/opacity, structures-only display and axial cutaway work. Switching to
the original T1 shows only its original SynthStrip brain layer; switching back
restores the tissue view's settings. A native save and a fresh process restore
the estimated source, layer visibility, opacities and cutaway, with all four
surfaces fully rebuilt. The [native interaction record](ui-demonstration.json)
hashes five local screenshot/accessibility pairs and the actual renderer build.

This is a display integration of the [bounded v3 fitting candidate](../poc-ixi265-atropos-typed-mask-run-v3/RESULT.md).
Tissue names remain unreviewed intensity assumptions, raw memberships are not
physical partial-volume fractions, and CSF-like intensity is not a ventricle
segmentation. This adds zero patients or observed surgical effects.

## Saved source and derived evidence

The canonical host, primary acquired T1, original estimated brain, primary case
and planning identities, 16 archived files and two reference displays remain
unchanged. One separate T1 estimate display is appended. Twenty-one additional
archived files preserve ten native result maps, three raw native posterior
files and eight provenance records: 37 files totaling 69,176,286 bytes. They are
source and derived evidence, not 37 original acquisitions. No inference runs
when opening, changing layers or saving.

The [sealed packaging outcome](outcome.json), SHA256
`e2d3f6b2c589a792fe62688fb035157e7d6ff770864bd597a61151e7c8cabdef`,
records one 11.732 s execution, 1,523,679,232 sampled supervisor/family RSS bytes
and at least 6,189,219,840 available bytes. Its 2 GiB/300 s/512 MiB output guards
pass; source/result inputs remain unchanged. The saved bundle is 113,750,604
bytes, SHA256 `bfa03b853dcb8e6893982e75a6c113e8ba3de1ce88225bb631f8a758ded1350f`,
under `outputs/poc-ixi265-tissue-display-v1/`. Expanded display attachments total
334,233,600 bytes within the existing 384 MiB limit.

The [independent package audit](../ixi265-tissue-display-independent-v1/independent-qa.json),
SHA256 `817b26109d53a89930caa5c8e80572f92b218e9d9b5dab4f15fd6b0622d14155`,
streams all archived bytes and checks primary/reference preservation, actual
native mask equality and exact acquired-T1 display arrays. Class counts remain
263,320 / 607,861 / 580,320, plus the exact eight UNKNOWN cells. Known and UNKNOWN
partition the original brain. No registration, planning, training, host or
anatomical acceptance is introduced. The audit peaks at 466,075,648 RSS bytes
and executes no native model or solver.

## Native save and verification

The GUI review copy lives under
`outputs/poc-first-qa/ixi265-tissue-ui/saved/IXI265-tissue-estimates-reviewed.ressectionlab`,
SHA256 `1cfb479807c92bb76537df0e188c161df5e6488aabeb19021c8ba72d0485c024`.
Its restored view selects the estimate display, hides CSF-like, retains gray/
white/UNKNOWN and uses opacity 0.12/0.26/0.70/1.00, structures-only and axial
cutaway. Both native processes exit zero. An independent saved-copy audit is
recorded after this actual interaction: [GUI-save audit](../ixi265-tissue-display-independent-v1/gui-save-independent-qa.json),
SHA256 `9167421a982a8b5852ee08e62065443b9a99764d1bc780341669004bc1fad351`.
It verifies all arrays, archived files and three display sources unchanged, plus
the selected source, visibility, opacity, volume and cutaway settings.

The [initial metadata guard finding](../ixi265-tissue-display-independent-v1/gui-save-initial-metadata-guard.json)
retains a checker rejection before any array decoding: ordinary native Save adds
an `artifacts.workspace` view record. The separately reviewed
[final guard preparation](../ixi265-tissue-display-independent-v1/gui-save-final-guard-preparation.json)
permits only that exact display-settings record with empty route selections.
Every source field and array check remains strict. The original checker, first
finding/preparation, earlier wrapper snapshot and package audit remain unchanged.
The successful saved-copy audit peaks at 464,683,008 RSS bytes. No product or
patient-source change was needed to correct this audit assumption.

Thirty generated/backend persistence controls, seven renderer source-choice
tests, TypeScript and production build pass. The optional readable source
caption retains mandatory model/source and registration status in the selector.
The prior fitting denominator remains three real protocol attempts, two failures
and one completed unaccepted numerical candidate. Anatomical accuracy, normal-host
review, registration, physical tissue fractions and tissue-conditioned growth
remain open. The twelve-source catalog and frozen nine/ten history are unchanged.
