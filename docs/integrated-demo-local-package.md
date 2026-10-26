# Local integrated demo package

`scripts/package_integrated_demo.py` freezes the complete tracked `src/` tree,
scripts and desktop sources from commit `d793e6925139bbec835acced6e7b9e25720e11ac`.
It copies the selected development data and evidence byte for byte into a
manifested local `tar.gz`. The archive is for local reproduction; it is not a
clinical artifact or an externally published release.

The bundle contains the ten pinned NIfTIs for fresh UCSF-PDGM-0005 and 0007,
their original source receipts and development declaration, the 0005 v4 access
case, the 0007 v1 fitting-source access case, the 0007 v4 learned case, BC and
PPO checkpoints, training and failed 0005 transfer receipts, and Electron
playback evidence. It includes installed SynthStrip and MRI_GBM model bytes,
configuration, acquisition hashes and license notices. The builder omits
unrelated datasets, model families and Python/Node runtimes.

Build a new ignored output directory:

```sh
.venv/bin/python scripts/package_integrated_demo.py build \
  --output testcases/outputs/integrated-goal-20261026/my-local-package
```

The output holds `bundle/manifest.json`, `bundle/VERIFY.md`, a compressed
archive, and its SHA-256 build receipt. `VERIFY.md` gives the exact extraction,
native replay and frozen PPO commands. Use a compatible same-host `.venv`:
the checkpoint binds Python, platform, numpy, scipy, torch, producer source,
and the authoritative engine fingerprint. The archive does not provision those
runtimes. Fresh-install SynthStrip or MRI_GBM inference and a new Electron
dependency installation are not established by the package verification.

The saved native scenes and learning receipts retain historical absolute
source paths as provenance. Relocated replay takes the bundled archives as
explicit inputs and never rewrites sealed source identities. The package
verifier checks every member's size and SHA-256, source NIfTI LFS receipts,
model installation receipts, native branch replay, and the frozen PPO endpoint.
The 0005 frozen transfer remains an incomplete failed result.
