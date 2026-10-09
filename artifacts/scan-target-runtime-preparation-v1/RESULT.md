# Scan-target runtime and checkpoint preparation

Completed 2026-10-09 UTC. The isolated CPU runtime and bounded checkpoint
inspection passed. Patient registration, network inference, segmentation
accuracy and planning remain untested in this slice.

## Runtime and numerical checks

The ignored `.tools/scan-target-runtime/venv` uses CPython 3.12.14 on macOS
arm64, with nnunetv2 2.5.2, antspyx 0.6.1, NumPy 2.0.1 and Torch 2.14.1.
The project `.venv` was unchanged. The
[platform-specific lock](../../requirements-scan-target-macos-arm64-lock.txt)
pins 86 distribution archives by SHA-256. Their reported total is 323,417,037
bytes, excluding small build-tool transfers. Build dependencies of source
distributions are not separately locked, so this is not a complete reproducible
build specification. Dependency consistency and CPU import checks passed.

A manufactured 40³ image registration check reduced RMSE from 0.064582 to
0.000561; inverse image resampling gave 0.000557. Known forward-point error was
0.003354 mm, and root independently reproduced the saved affine point round trip
at 0.00000213 mm. The initial inverse check **failed** at RMSE 0.109062:
explicit `whichtoinvert=[True]` was required for the single affine matrix.
That failure is preserved. These are numerical API checks, not patient evidence.
The [fixed atlas](../sri24-atlas-intake-v1/RESULT.md) has shape
(240,240,155,1); only its singleton fourth axis was removed, retaining its affine.

## Fixed weights and safe metadata inspection

The separately licensed [GlioMODA v1.0.2 weights](https://zenodo.org/records/15625233)
are CC-BY-4.0. The completed archive is 3,306,003,582 bytes, publisher MD5
`428d96a19480ecfa391b3cdd5a7a535b`, independently reproduced SHA-256
`067ea69d5fb4d6a000aa7cda0396380a4212b05bce85b846da43eefc09c13855`.
The GlioMODA wrapper was not installed or imported; its code-license question
remains separate from the weights. An independent nnU-Net adapter is required.

All 120 ZIP members passed path, duplicate, symlink and encryption checks.
Only the selected dataset metadata (727 bytes), plans (10,802 bytes) and
checkpoint (250,184,062 bytes) were extracted under the ignored
`data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1/` directory. Checkpoint SHA-256:
`0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`.

The first CPU `weights_only=True` load failed on NumPy scalar metadata. Static
review in the actual NumPy 2.0.1 runtime supported a four-entry scoped allowlist:
the scalar with its historic module alias, `numpy.dtype`, and float32/float64
dtype classes. One subsequent bounded safe load succeeded, restored the prior
safe-global state, and reproduced the checkpoint hash before and after. No
unrestricted pickle load was used. Peak sampled memory was 480 MiB, below the
3 GiB cap.

Metadata declares `nnUNetTrainer`, `3d_fullres`, 1 mm spacing, 128³ patches,
input channels 0=T1c and 1=T2-FLAIR, and three region outputs. Whole tumor uses
labels {1,2,3}, core {1,3}, enhancing tumor {3}; region class order is [2,1,3].
All 292 network state tensors were finite. Their 88,624,655 entries include
aliases and **are not a count of unique model parameters**. Full network
construction and a forward pass have not yet run.

## Evidence and remaining dependencies

Local evidence is retained under `.tools/scan-target-runtime/`,
`build/limited-input-guard-design/gliomoda-weights-v1.0.2/` and
`build/scan-target-estimator-independent/`. The install lock SHA-256 is
`f78bda5e88157d0974a81b53c08f79895841db4e48934aeb4058b08f31c5fde1`.
The analytic harness is an ignored experiment record, not yet a portable
patient-processing CLI. Neither weights nor patient arrays are committed.

Next is an independently authored adapter with scan-only preprocessing and
native-frame inverse QC. Case4 remains DEVELOPMENT: local alignment and known
inferior/cerebellar skull-strip omissions are unresolved. Pretraining overlap
must be audited before any held-out claim. No private patient annotation,
patient registration, patient inference, route evaluation or RL run occurred
in this preparation slice.
