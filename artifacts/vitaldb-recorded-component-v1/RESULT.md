# VitalDB recorded pump and cuff-pressure component replay

**Implemented and independently checked on one real DEVELOPMENT person:** local-file-only numeric import, exact native records, deterministic prefix replay, record-age inspection, and bound snapshot reopening. This is an API/CLI component; desktop integration and the full physiology goal remain unfinished.

The original seven-track arterial request **failed**. The authenticated PhysioNet 1.0.0 file has none of `Solar8000/ART_MBP`, `ART_SBP`, or `ART_DBP`; its pump-rate unit is `mL/h`, not the live inventory spelling `mL/hr`. The original declaration and reproducible header-only refusal remain unchanged. A separate declaration, frozen after native-header inspection and before selected values, admits the four exact Orchestra tracks plus explicitly named `Solar8000/NIBP_MBP`, `NIBP_SBP`, and `NIBP_DBP`. NIBP is never aliased to arterial pressure.

## Actual evidence

Case3 / subject2861 remains DEVELOPMENT; all other VitalDB people remain unopened and unassigned. Selection used inventory/header availability, not outcomes. This adds one component-development person, zero training examples, zero optimizer updates, and zero RL transitions.

The official mirror delivered bytes matching the original release's published checksum:

- Recording: 6,537,712 bytes, SHA256 `573db0941d580167833f84497f5d1c4a1391443eeec3f95852287ea09db024e4`.
- Pinned parser: VitalDB 1.7.2 MIT archive, 67,723 bytes, SHA256 `70e0ce784b13d52bbf6a315b742ab06d5ed0599b8b01f76182971a391dd64130`. Retained `utils.py` SHA256 `b0e88b1365c8d9a88814123c9a5b5a5d3c696a5c4849c7db3ebda476cc7134e3`; no dependency installation or upstream execution.
- Production reader SHA256 `ccd92d0e36126f4b4c2b53325337b21e03c3e7861bdc43263b7acffd4b3da1be`. All executable/declaration hashes appear in [verification.json](verification.json).

The source-front-end attempt timed out after 120.027 seconds with 1,310,684 partial bytes; these are retained and excluded. A separately declared single official AWS-mirror attempt completed in 15.585 seconds with the complete published hash. Both actual acquisition-code snapshots and receipts remain retained. Acquisition refuses automatic retries, redirects, curl configuration files, and overwrites.

The full gzip CRC/EOF check consumed 24,975,256 decompressed bytes and 298,661 packets; largest packet 216 bytes. There are 81 native track headers. Only the seven declared numeric tracks are decoded; 19,806,236 unrelated payload bytes are skipped without value decoding. Bounds are 6,537,712 compressed bytes, 128 MiB decompressed, 4 MiB per packet, one million packets, and 250,000 selected records.

| Native channel | Records | Maximum source-record gap (s) |
|---|---:|---:|
| Orchestra/PPF20_RATE | 4,103 | 269.290 |
| Orchestra/PPF20_VOL | 4,103 | 269.290 |
| Orchestra/RFTN20_RATE | 4,086 | 286.312 |
| Orchestra/RFTN20_VOL | 4,086 | 286.312 |
| Solar8000/NIBP_MBP | 1,864 | 19.830 |
| Solar8000/NIBP_SBP | 1,864 | 19.830 |
| Solar8000/NIBP_DBP | 1,864 | 19.830 |

All 21,970 records retain original timestamp/value bits, native units, track and packet ordinals. Each selected track has zero duplicate timestamps, out-of-order records, and nonfinite values in this actual file. Across tracks there are 7,207 distinct timestamp boundaries. Missing intervals remain unfilled; equal values are not deduplicated into clinical events. The parser supports explicit nonfinite scalar representation (`null`, original bits, and kind), but this file supplies no positive nonfinite example.

## Clock and measurement limitations

The native v3 header has only 10 bytes and lacks optional start/end timestamps. Selected native timestamps span `4102444802.1943`–`4102449193.2004004`, contradicting the initial assumption that these native bytes already use release-relative seconds. The initial QC interpretation is retained as **superseded and invalid** in `qc-before-clock-review.json`; [the clock review](../../manifests/vitaldb-native-clock-v1.json) supersedes only that interpretation. The possible 2100 epoch suggested by the parser's anonymization default is not accepted as this recording's origin.

Accepted replay uses `unmapped_native_source_seconds`, null recording-start origin, and identity mapping of raw timestamps. No offset is subtracted, no episode elapsed time or actual acquisition date is asserted. Snapshots bind this correction, source/person/role, code, component declaration, and authenticated legal/parser evidence. Whole-case QC stays outside replay frames; records are released only at or before the supplied native clock.

`source_record_age_seconds` means age of the published device record. Monitors may repeat a previous cuff result, so actual measurement age stays unknown even when a record is published exactly at the replay clock. Device latency and synchronization error are unavailable. Pump names may have been assigned retrospectively; pump values do not establish administration actions, delivered dose, bolus timing, or causal response. The source release also removed all-zero and fewer-than-ten-sample tracks. No continuous physiology, clinical dosing, glioma experience, causal effect, or RL eligibility is established.

## Verification and runnable commands

Focused checks: **15 passed, zero failed/skipped** (seven pure format/refusal controls and eight actual-source tests). [Independent comparison](implementation-comparison.json) matched all seven headers and every selected record, checked all 7,207 event boundaries for exact prefixes and record ages, reproduced snapshot continuation/full-prefix digest, and refused eleven changed snapshot bindings. The independent source files are retained beside the report.

For a fresh checkout, prepare the exact ignored cache locations first. This separate reproduction command uses Python's standard library and system curl; it does not install or execute the upstream parser:

```sh
python3 scripts/prepare_vitaldb_cache.py
python3 scripts/prepare_vitaldb_cache.py --offline
```

Existing files must pass exact size/SHA256 checks; corruption refuses without repair. Only missing pinned parser/archive bytes (67,723 bytes, 120 seconds) and the same official-mirror case3 recording (6,537,712 bytes, 300 seconds) can be downloaded. Only authenticated `utils.py` is extracted, with a 1 MiB member bound. Each local artifact gets one durable attempt marker in ignored `build/vitaldb-cache-preparation-v1`; failures retain partials and receipts and cannot retry automatically. Historical study declarations, acquisition receipts, legal notices, and reader code remain unchanged. This reproduces the same DEVELOPMENT person and adds zero people or training contributions.

[Cache verification](cache-preparation-verification-v1.json) records 16 passing controls, including the actual current cache with networking disabled and isolated missing-cache tests using already authenticated source bytes through mocked transport. The actual offline invocation made zero downloads; no new remote cold-download success is claimed.

[Independent cache review](cache-preparation-independent-review-v1.json) passed those controls and admitted the isolated freshly prepared caches through the unchanged reader: all 21,970 records, source bindings, null origin, and unmapped native clock were preserved. Its second offline preparation made zero network attempts or downloads.

Run the observed component checks from the repository root after preparation. No command downloads another patient or creates generated observations:

```sh
.venv/bin/python -m pytest -q tests/test_vitaldb_observed.py
.venv/bin/python scripts/replay_vitaldb_component.py qc
.venv/bin/python scripts/replay_vitaldb_component.py original-check
.venv/bin/python scripts/replay_vitaldb_component.py replay --clock 4102444900
.venv/bin/python scripts/replay_vitaldb_component.py replay --clock 4102444902 --reopen artifacts/vitaldb-recorded-component-v1/replay-snapshot.json
mkdir -p build/vitaldb-independent-reproduction
cp artifacts/vitaldb-recorded-component-v1/independent_reader.py artifacts/vitaldb-recorded-component-v1/compare_implementation.py build/vitaldb-independent-reproduction/
.venv/bin/python build/vitaldb-independent-reproduction/compare_implementation.py
```

`original-check` deliberately exits 2. QC is whole-case retrospective inspection and must not be supplied as a deployment input. Replay clocks are native source seconds. Optional `--snapshot` and `--output` require distinct, previously absent files in the permitted artifact/output directories; source aliases and existing/protected destinations are refused. Missing local source/parser caches cause replay refusal; use the separate preparation command above, not the historical one-shot acquisition runner. An earlier failed local cache-preparation attempt remains a refusal requiring explicit review, not a reason to delete its marker or retry automatically.

Source attribution: Lee H, Jung C (2022), [VitalDB v1.0.0, PhysioNet](https://physionet.org/content/vitaldb/1.0.0/), DOI [10.13026/czw8-9p62](https://doi.org/10.13026/czw8-9p62), CC BY 4.0. Original publication: Lee HC et al., [Scientific Data 9, 279](https://doi.org/10.1038/s41597-022-01411-5). Exact release license and complete upstream MIT notice are retained; changes comprise this strict local numeric reader and replay interface.
