# Reproducing reports and historical audits

This note records a static inspection of tracked files, ignored paths, entry
points and recorded source identities on October 4, 2026. No installation,
training, application launch, test run or audit execution was performed for this
inspection. The current Electron development workflow and synthetic fixture use
the default Python lock without Qt or VTK; see [dependency validation](dependency-lock-separation.md).

The historical studies below have different requirements for regenerating a
report and rerunning the full saved-artifact audit. Paths are relative to the
repository root. In this note, `STUDY` means
`artifacts/learning/procedural-native-to-ucsf-v2` and `REPLAY` means
`STUDY/comparison/development-geometry-e8a2cc46-0d88-4fc3-93be-176711ee19aa`.

## Report generation from tracked evidence

`STUDY/report.py` already reads either JSON or its `.json.gz` counterpart. All
eight evidence inputs it reads are tracked, including
`REPLAY/native-history-replay.json.gz`; the uncompressed replay is ignored.
With the documented Python environment, the entry point is:

```sh
.venv/bin/python artifacts/learning/procedural-native-to-ucsf-v2/report.py
```

Run this in a disposable checkout or copy when preserving existing outputs: it
writes `RESULT.md`, `comparison.png`, `comparison.pdf` and `report-source.json`
beside itself. It summarizes saved records and does not reconstruct the original
training or repeat independent geometry evaluation. Its source checksums use
decompressed bytes. The existing `STUDY/report-reproduction-check.json` records
a successful historical regeneration from a copy containing only the gzip
replay, with equal Markdown and uncompressed source hashes. This inspection did
not repeat that execution.

`REPLAY/replay-storage.json` records both storage identities: the original JSON
is 28,053,557 bytes with SHA-256
`ef3b13a975eb979e4b9ae23b0afa8ab496128bdb4626cd14a0612a1132e9c41b`;
the tracked gzip is 470,315 bytes with SHA-256
`19f8317561e766f652d4d6bd0afd7fcdafc01d28478fc664ff2f2507c4cf561b`.

## Original audit prerequisites

The executed audit programs and their original receipts remain unchanged:

| Original entry point | Inputs absent from an ordinary clone |
| --- | --- |
| `artifacts/learning/procedural-native-to-ucsf-v2-independent-audit/audit.py` | `STUDY/frozen-source/`, `STUDY/source-case.ressectionlab`, the raw `REPLAY/native-history-replay.json`, and the checkpoint files listed below |
| `artifacts/native-frontier-expansion-v1/independent-audit/audit.py` | `STUDY/frozen-source/`, `STUDY/source-case.ressectionlab`, and the raw prior `REPLAY/native-history-replay.json` |

Both original audit helpers read the prior replay with `Path.read_text()` and
have no gzip fallback for that file. The frontier audit does correctly read and
hash-check its three tracked expanded-candidate gzip files under
`artifacts/native-frontier-expansion-v1/experiment-2/`; that support does not
cover the prior fixed-inventory replay. Restoring the raw replay by losslessly
decompressing its tracked gzip, with both sizes and hashes checked against
`replay-storage.json`, addresses only this storage prerequisite.

The learning audit also reads these ignored tensors under `STUDY`:

- `pretraining/procedural.pt` and `comparison/procedural-source.pt`;
- `pretraining/training/initial.pt` and `pretraining/training/checkpoint.pt`;
- `comparison/scratch-{11,23,47}/initial.pt` and `checkpoint.pt`;
- `comparison/adapted-{11,23,47}/initial.pt`, `checkpoint.pt` and `procedural-source.pt`.

An archived case and checkpoints must be supplied and checked against the
recorded bundle, semantic, planning and tensor identities. Hashes verify supplied
bytes; they cannot recover missing case data or model tensors. Reacquiring and
preparing a current case is not proof of byte identity with the archived bundle.

## Source reconstruction and scope

`STUDY/worker-source.json` records numerical source commit
`68e4fde13b1fa13411e59af663bd17ae63885947` and numerical runtime identity
`sha256:e1186e12e79263c41e3cdad06112012702b4126a73fd912c14fa1e41d6d8843f`.
The historical learning audit checks numerical files against that commit using
`git show`, so its Git objects must be available, including in a shallow clone.
The scientific declaration at
`manifests/experiments/procedural-native-to-ucsf-v1.json` is additionally checked
against commit `f26a72c30ba356c9cb387ee87fe98a69d7a27ffa`.

`STUDY/launch-source.json` records no Git revision. The recorded worker tree
also contains dirty paths. Therefore checking out the numerical commit alone
does not establish that all 138 files required by the learning audit equal the
launch snapshot. Restore the archived `frozen-source/` tree, or reconstruct each
available file and verify every `file_sha256` and `numerical_runtime_sha256`
entry. The frontier audit requires the recorded numerical subset from that same
snapshot. These source manifests remain the authority over today's source tree.

The frontier audit separately binds
`artifacts/native-frontier-expansion-v1/experiment-2/experiment-script.py` to
its declaration's script SHA-256
`6c1536b093b26bbdd611bc0478f48ad5f2c762b8a48e2acf1779e2984238f75f`
and requires equality with `scripts/probe_native_frontier.py`. A later change
to the current script must not silently replace the executed version.

These requirements supersede the frontier audit README's original claim that
reading compressed histories alone makes the complete audit work from a clean
clone. The saved receipts describe the execution that actually occurred with
local archived inputs. Full reconstruction from a new clone has not been
demonstrated here. Any later reconstruction should write a new receipt and
preserve the original scripts, reports and experiment outcomes.
