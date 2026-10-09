# Independent generated streaming profile launcher review

Decision: **GO for a separate root-issued, one-attempt generated profile release using the exact frozen bundle below.** This source-only review does not authorize or report a realistic-size run. The actual profile output directory is absent. No model, patient payload, native solver, training, or network operation was performed by this review.

Observed source commit: `abbc3e31ef2dd9123e470a31b3701eb375f85424`. All reviewer writes are in this ignored directory; no tracked edits or commits were made.

## Frozen release inputs

All paths below are relative to the repository root.

| File | SHA-256 |
| --- | --- |
| `build/private-vascular-streaming-preparation-v1/launch_profile.py` | `e48cb01e2586e40776109bfa0e79ad10b6e18ab0e6771c3642b9b46d0f6a5480` |
| `build/private-vascular-streaming-preparation-v1/test_launch_profile.py` | `f66a9f813793cf4fcd9791c8f406e605c66833753619d5c1e5c5753db9093ca3` |
| `build/private-vascular-streaming-preparation-v1/streaming_contact.py` | `c85cca298b20190be865e23cb3d0bde3dc142e69f182e93e65c008895cec3d75` |
| `build/private-vascular-streaming-preparation-v1/profile_generated.py` | `1b149489ee2dfb0338c08518fba5835b689de3cace7f696796b08b42e27b4eda` |
| `build/private-vascular-streaming-preparation-v1/repository-source-pins.json` | `2ced65d6f47c3771919e6a62f9fbbf54593db30f803c70343b798db83513fea8` |
| `build/private-vascular-streaming-preparation-v1/prior-repository-source-pins-v1.json` | `264f0e811dbb117357de979c2748ac687b7108e56e1f183d8c6582f4b4a1861d` |
| `build/private-vascular-streaming-preparation-v1/source-rebind-v2.json` | `f816fbdb334a1b519ef506c2c68f8389dfb0d4cc43c2aa08116664bfce3f57df` |
| `build/private-vascular-streaming-preparation-v1/supervisor-disabled-release.json` | `a438d23e84e2a414d83c9ad7ec6d3f590f40414c3251d573b002234d05ea6951` |
| `build/private-vascular-streaming-independent-v1/REPORT.md` | `12d26ca020626ebc0f8ae3474161b2a8cf641610e0714afb63fbb721bf78d488` |
| `build/menichetti-structure-launch-preparation-v1/launch.py` | `2f93c7fd776014417677ed7ea12112a8f237218e76d288ad04297b651816fea2` |
| `build/menichetti-structure-launch-independent-v1/REPORT.md` | `7b502043e15b8a402c1d02e2bc1338e50cbc201977af6297afd01c5af64731ff` |
| `scripts/mechanics_hbe_v5_remaining_one_shot.py` | `90d855b1e9518793480688afbfb133b586a6ccbe5b974be30b6b40624336216b` |
| `scripts/mechanics_hbe_v5_n8_one_shot.py` | `82bb01dc2588f750489916c789e7afe85b7a9a6e1ef0fcd5820fe8e70be4f3a0` |
| `scripts/febio_runtime.py` | `679594d7f3759f5485b9fb868e7e5543ebb242bccd24d112f9ca862bb6d2a01e` |

The three tracked supervisor/runtime helpers are also checked against Git blobs at `0f7eee3a17b6743fe15549cf8aba396fa4e98a2c`. This historical binding deliberately does not claim that today's HEAD equals that historical commit.

## Source inventory rebind

Independently compared the old 76-source and new 77-source JSON inventories, verified every current source hash, and checked the exact declared delta. Only these paths differ:

- Added `src/resectionlab/matched_private_vascular.py`, SHA `148c30d147c4f88f0e725575debb8b50b774e224e80d4b5bf9b18d90465945db`.
- `src/resectionlab/private_vascular_evaluation.py` changed from `30e9efaac9d0f4e311bedf9075362fcd488f1b31f76a3938f4224736d39e2eb7` to `a4ca5a710fa6b8f02251e7d30d5e3fc6bf3602a8f6b2ca9ce7af6e809281856d`.

Inspected the root integration diff. These changes concern matched-strategy preflight/private scoring; the streaming kernel, profile worker, independent geometry query, evaluation module and other source hashes remain unchanged. They are not imported by this generated profile's geometry path. The prior tiny geometry review remains applicable without repeating kernel tests. This review does not independently approve the new matched-strategy implementation itself.

## Supervisor findings

No blocking findings remain for the declared generated profile scope.

The launcher uses the exact existing `OwnedWorker`, existing stage supervision and numeric process-group RSS observer. It retains the child handle, allows one child, writes a reserved receipt before spawning, and always attempts owned cleanup after supervision. A group-signal failure does not skip the direct-child kill/reap fallback. Any cleanup fallback, error, absent containment or unsuccessful stage prevents success. The reviewed negative paths preserve failed receipts and outputs; there is no retry or cap escalation.

The immutable cap contract is one attempt; 35 seconds for the child stage; 10 seconds reserved for cleanup within a 45-second accepted lifecycle; sampled process-group RSS 512 MiB; aggregate output 4 MiB; profile JSON 1 MiB; numerical thread settings 1. The kernel's existing cooperative work and 20-second wall limits are unchanged. The worker's imports and execution occur inside child stage supervision. Preflight source checks occur before that stage clock.

Final acceptance requires the exact output inventory and unchanged hashes for console, child release, worker attempt and profile; unchanged launcher/root release and source pins; exact persisted receipt bytes equal to the in-memory result; and remaining lifecycle time. A late console, attempt, profile, receipt, release or launcher change, or an extra file, cannot produce a passing final receipt in the tested boundaries. The final inventory permits only receipt, console, child release and a worker directory containing attempt/profile JSON.

The semantic check goes beyond exit zero: it checks source/origin bindings against the current manifest, fixed grid/action/capsule scope, complete work counters and maxima, sampled union count, count partitions and union bounds, finite nonnegative times/memory, duplicate action equality, derived encounter/coverage flags, null clinical claims and count-derived 0.7 mm cell volumes. It accepts only the generated contact result; removal evaluation and strategy/patient admission remain false or not evaluated.

## Independent execution evidence

Command used from the repository root:

```text
.venv/bin/python -B build/private-vascular-streaming-launch-independent-v1/run_review.py
```

The runner records the complete absolute pytest command and thread environment construction in its source/receipt. It uses a fresh ignored bytecode prefix, disabled pytest plugin autoload/cache provider, separate ignored temporary files, a 30-second outer limit and group termination if that outer limit expires.

Result: **45 passed in 1.08 seconds**, exit 0; elapsed enclosing test execution 1.425625 seconds. This includes 28 author controls and 17 independent controls. All **93** bound file snapshots were identical before and after. No outer timeout occurred.

The additional controls reject ten contradictory semantic summaries; reject late console, worker-attempt, root-release and launcher mutations; verify second-attempt refusal before worker import; and inject a parent exception after an actual tiny Python child starts, verifying fallback termination, direct-child reaping and an empty remaining process group. The source rebind and absence of the actual profile directory are checked. The author controls additionally exercise actual tiny wall/RSS/output refusals and group-signal PermissionError cleanup. Real child commands are restricted to short Python `-c` fixtures; no realistic profile worker is launched by this suite. Generated successful summaries are test fixtures, not measured performance results.

The exact false-release command also passed separately:

```text
.venv/bin/python -B -X pycache_prefix=/Users/sayemkamal/.codex/.chatgpt-projects/g-p-6ac1ef0d1fc481919c65bf30beca2549/RessectionLab/build/private-vascular-streaming-preparation-v1/launcher-unused-pycache build/private-vascular-streaming-preparation-v1/launch_profile.py --release build/private-vascular-streaming-preparation-v1/supervisor-disabled-release.json --release-sha256 a438d23e84e2a414d83c9ad7ec6d3f590f40414c3251d573b002234d05ea6951 --check-only
```

It returned `bindings_valid`, `execution_released: false`, and the exact caps above, with no worker output directory created.

| Reviewer evidence | SHA-256 |
| --- | --- |
| `test_independent.py` | `2e32d0acfd68c9bbf9c844676acb65ed6f1c1ab13f8a5477368dc790e2398888` |
| `run_review.py` | `316bf97c1d9a24adc45e2e793b09667b2894bd8162f1f59520a139ba6f9ee8f5` |
| `run-receipt.json` | `24f444f009146e51c75e4417903bac16b4fa0d478303645018ec6f898850ac20` |
| `pytest-output.txt` | `124d710b4e634f773c0cfb2b7fa4c382cf3ef92f8066c66d3ce7bb67d6aef790` |
| `source-before.json` and `source-after.json` | `688253554fc31fa06f7d92e1634f997c14490757ed1be321d8f33c7635d5dc55` |
| `check-only.txt` | `8ed321498617b9099d38c204b41793aac210995fe49a72ea1e7937fca46d3fce` |

## Root release conditions and limits

Root must verify this report and the exact frozen executing hashes before issuing its release. The release schema does not itself contain this launcher-review hash; binding this report is an explicit root preflight obligation. Use the frozen disabled release as the template, changing only `execution_released` to true, then hash the exact resulting release bytes. Keep all source bindings, caps, output directory and false patient/model permission unchanged. Invoke the launcher with the same absolute parent cache prefix shown above, the new release path/hash, and without `--check-only`. Do not launch the profile worker directly. Both the fixed output path and parent cache prefix must still be absent; preserve any failed first attempt and do not retry.

RSS and output are sampled; brief peaks can be missed. The lifecycle limit is an acceptance/cleanup deadline enforced by the cooperating parent, not a separate operating-system watchdog for that parent. The recorded `lifecycle_seconds` is captured before the last evidence checks; acceptance rechecks the clock after them. Full scientific runtime/binary closure is not authenticated by this feasibility bundle. The kernel has no patient loader, model inference, clinical safety/injury interpretation, removal-overlap evaluation or complete strategy admission. Passing tiny controls does not prove realistic-grid feasibility; only root's separately released measured attempt can provide that evidence.
