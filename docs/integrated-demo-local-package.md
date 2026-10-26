# Local integrated demo package

## Current v6 research package

`scripts/package_current_integrated_demo.py` freezes complete tracked source,
scripts and desktop files from a named commit. The current package uses
`12af2fb8185af4ed54f21c57d18802ba59c8cbe2`, including the native engine,
model producer, Electron frozen execution action, and cancellation and admission
rollback UI. It carries 0007 novelty-selected access v4, v6 BC and PPO
checkpoints and learned native branches, and 0005 access v6 with both abstained
frozen transfer branches. Both real Electron executions are present with their
input and output native archives, logs, receipts, saved workspaces, and images.
The original five pinned NIfTIs per case, DEVELOPMENT declaration, provenance
receipts, installed SynthStrip and MRI_GBM bytes, and license notices are also
included. Earlier v4 evidence remains a separate historical package.

Build the current package in a new ignored directory:

```sh
.venv/bin/python scripts/package_current_integrated_demo.py build \
  --commit 12af2fb8185af4ed54f21c57d18802ba59c8cbe2 \
  --output testcases/outputs/integrated-goal-20261026/reproduction-package-current-v1
```

The archive's `VERIFY.md` gives exact manifest, native replay,
positive 0007 frozen PPO, and negative 0005 frozen PPO commands. The latter
remains an abstained outcome with a failed target gate. Verification runs from
the extracted package source, checks exact endpoint states and unchanged input
hashes, and requires zero optimizer updates. It requires the matching same-host
Python runtime; the package does not provision Python, Node, Electron, or model
inference runtimes. Fresh-install inference and a new Electron installation
have not been verified. Sealed historical absolute paths in receipts remain
informational provenance and are not rewritten.

For example, extract the archive into a new ignored directory, then use its
own verifier and `VERIFY.md`:

```sh
.venv/bin/python scripts/package_current_integrated_demo.py extract \
  --archive testcases/outputs/integrated-goal-20261026/reproduction-package-current-v1/integrated-current-local.tar.gz \
  --output testcases/outputs/integrated-goal-20261026/reproduction-package-current-relocated-v1
cd testcases/outputs/integrated-goal-20261026/reproduction-package-current-relocated-v1
/absolute/path/to/matching/.venv/bin/python scripts/package_current_integrated_demo.py verify-native --bundle .
```

## Historical v4 research package

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
