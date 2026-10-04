# Fixed main-model support inference on four real BTC scans

All four declared main-only SynthStrip runs completed successfully in 35.023912 seconds on October 4, 2026. Each ran once, in the fixed PAT22 → PAT25 → PAT26 → PAT27 order, with no failures, retries, model downloads, package installs, or model training. PAT22/PAT25 retain population-training roles; PAT26/PAT27 retain checkpoint-selection development roles. PAT29/PAT31 and UPenn were not used.

| Subject | Child inference seconds | Main mask SHA-256 |
| --- | ---: | --- |
| PAT22 | 7.901334 | `be7440cdba7fc921cf32615fa632ebc58858bf205bdc688746540dfd73e81591` |
| PAT25 | 6.847232 | `70b2b1bfe1234a32946e82ac6525f6c6cbfeefa074d6c340ab0e27f5efce3fc7` |
| PAT26 | 6.073117 | `80aa8b90713525c8b95ded63a190113a0f3c50260cc08241f2344f030249b58d` |
| PAT27 | 6.322828 | `4b990b4dda738ed14565624e4e190db8b0145e15fd537ddffb24a36e94811709` |

The original PAT05 wrapper, main checkpoint, upstream script, and retained license matched their declared hashes. The unchanged `run_synthstrip(model="main")` function ran from committed source `ea501a8991e9a4ec6a6154f117c48a8b6ff1e390`, with the same 1-mm border, two CPU helper threads, MPS adapter, 300-second child timeout, and 6-GiB bounds. No-CSF inference and the unused intensity-baseline comparison were omitted. All source cases, T1s, model assets, and 91 frozen source files remained unchanged. Runtime versions matched the declaration; historical PAT05 interpreter/package binary equivalence is not attested.

No tumor annotation or source case array was opened in this inference stage. All four model outputs were fixed before annotation-overlap QC, so selection-case targets could not choose settings, variants, retries, or access. The wrapper checked binary/nonempty mask and finite predicted-distance output on each native grid. Independent array/visual QC and proposal-bundle integration are pending in a separately scheduled stage. Source cases remain unchanged with no working brain mask, review promotion, or cortical access.

Maximum sampled process RSS was 678,838,272 bytes. Each run reported sampled Metal tensor allocation of 4,117,935,104 bytes and driver allocation of 5,948,243,968 bytes. These samples can miss transient peaks; process RSS omits GPU accounting. Predicted distances are model outputs, and their exterior 100-mm fill is not a calibrated clearance.

The prospective declaration is `manifests/experiments/brain-extraction-btc-spatial-main-v1.json`, SHA-256 `f6560e7d35a3c85d3e56f2abfbaa487337f22405a4be20dc28ddeb368f78450f`. `root-release.json` was written before inference. `inference-batch.json` preserves each attempt and every retained output hash; its SHA-256 is `57b41c12d937358165ca4aa8e46d0085b9577ee1619034a375e2c90d84bbc975`. `inference-summary.json` contains physical-frame/QC metadata, model rights and runtime records, original command, source archive identity, driver hash, and readiness limits. `execution.log` is the unchanged parent log.

`execution-driver.py` is a byte-identical tracked copy of the executed small driver. The original was invoked with the existing app interpreter, `-I -S -B`, the ignored driver path, and the absolute repository root. It then imported only frozen source plus existing dependencies. The full source archive remains outside Git at `build/btc-spatial-support-ea501a8/source.tar`; it can be reconstructed from the committed paths recorded in the root release. Native imaging, model outputs, logs, adapter, license, and reports remain under `outputs/brain-extraction/BTC-spatial-main-v1/sub-PAT##/`. Absolute historical paths and refusal to reuse an existing receipt intentionally preserve this run; any reproduction needs its own declared destination and release, with the original evidence unchanged.

These estimates derive from real public T1 scans. They are not reviewed brain/cortex anatomy, patient-specific functional evidence, clinical probabilities, or proof of surgical eligibility. Exact released-model training-patient overlap remains unverified.
