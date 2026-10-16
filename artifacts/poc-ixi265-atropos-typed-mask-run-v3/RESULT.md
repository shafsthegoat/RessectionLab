# IXI265 typed-mask fitting completes as an unaccepted estimate

One separately released native T1 run completes N4 and Atropos without changing
the acquired image, original estimated brain mask, fitting parameters or support.
The [sealed outcome](outcome.json), SHA256
`a9c241064f0ebc67e460c2f500f6b389e23bdb09fbda3667ab41d02b544c1bcf`,
records 19.126 s, 1,642,381,312 sampled supervisor/family RSS bytes, at least
6,968,803,328 available bytes and 47,245,684 output/temporary bytes. No declared
3 GiB/600 s/512 MiB guard fires. The process exits zero; no automatic retry runs.

The original brain support has 1,451,509 voxels. The exact eight zero-intensity
cells remain UNKNOWN, leaving 1,451,501 fitting voxels. The ordered classes have
263,320 / 607,861 / 580,320 cells. Their CSF-like, gray-matter-like and
white-matter-like names are intensity assumptions pending anatomical review;
CSF-like intensity is not a ventricle segmentation. Raw posterior memberships
are retained without normalization and are not physical tissue fractions.

## Independent saved-result evidence

The [independent audit](../ixi265-typed-mask-independent-v3/result-qa.json),
SHA256 `0b2918d3d6d1c17e772cad4649689f67ff9546083eced8bfe7d5796db0ed6509`,
checks all ten native outputs against the source T1/brain and hashes the host
bundle without decoding it. It independently reproduces the exact known/UNKNOWN
partition, class counts and hard-mask equality. All three posterior arrays equal
the retained native temporary outputs, remain finite within [0,1], are zero
outside fitting support and sum within 1.4156103134155273e-7 of one inside it.
All saved native shapes, selected affines, qform/sform codes and eight voxel-cell
corners pass the declared 1e-4 mm tolerance. Bitwise affine equality is not claimed.
The independent audit peaks at 502,579,200 RSS bytes and imports no ANTs/model.

The [exact generated-kernel preflight](../poc-ants-typed-mask-native-preflight-v3/outcome.json)
preceded this release. The [initial supervisor findings](../ixi265-typed-mask-independent-v3/supervisor-initial-findings.json)
retain two pre-fit guard defects and their counterexamples. The separate
[guard follow-up](../poc-ixi265-atropos-typed-mask-runner-guard-followup-v3/preparation.json)
and [independent supervisor audit](../ixi265-typed-mask-independent-v3/supervisor-qa.json)
verify the repaired cleanup/continuity path before the patient run. Initial
source snapshots remain unchanged.

## Preserved failures and remaining work

There are three real protocol attempts: two retained failures and one completed
numerical candidate. [V1](../poc-ixi265-atropos-run-v1/outcome.json) rejects eight
zero-intensity masked cells before fitting. [V2](../poc-ixi265-atropos-positive-domain-run-v2/RESULT.md)
preserves those cells as UNKNOWN but crashes at native N4 entry. The
[generated dtype comparison](../poc-ants-mask-dtype-generated-v1/RESULT.md)
establishes an adapter mask-type defect; this successful successor does not
symbolicate the earlier crash or establish its precise cause. Three earlier
SynthSeg memory failures remain separate evidence.

All three source files, runtime/lineage records, historical outcomes and attempt
markers are unchanged. Anatomical accuracy, individual tumor-free host review,
ventricles, physical tissue fractions and tissue-conditioned growth remain open.
This result accepts no anatomy, host, training or clinical use. Separate display
packaging and actual Electron inspection/save/reopen are the next integration.
