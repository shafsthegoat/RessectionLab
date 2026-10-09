# Independent saved-result audit: generated streaming contact profile

Decision: **PASS for the saved one-attempt generated contact feasibility result.** The released run completed within its declared caps with clean child reaping and consistent saved evidence. This does not admit patient data, clinical claims, complete-strategy evaluation, or an additional run.

This review read saved JSON/log/source bytes only. The pure-standard-library auditor performed no grid/contact evaluation, scientific import, child launch, native solver/model run, or network operation. All reviewer writes are in this ignored directory; originals remain unchanged. The audit covered **96 source/evidence files**, with identical hashes on its second read.

## Release, provenance and output integrity

Root's preflight binds independent launcher review `cf47706a02860411205dacf6f830a4da09081c10e07ef1c7eb6c01df76375af3` and source commit `abbc3e31ef2dd9123e470a31b3701eb375f85424`. The exact root release differs from the reviewed disabled template only by `execution_released: true`. The source/helper bindings, caps, output directory and false patient/model permission are unchanged.

Verified current launcher SHA `e48cb01e2586e40776109bfa0e79ad10b6e18ab0e6771c3642b9b46d0f6a5480`; all nine receipt source bindings; all 77 manifest source hashes; the exact source inventory; and the root/child release chain. The worker's recorded import origins are the pinned `resectionlab` package and independent geometry module. Source/cache checks pass and both fresh cache paths remain absent. Historical helper Git provenance was established by the frozen launcher review; this saved audit did not run Git or any subprocess.

The saved receipt has the expected canonical JSON bytes. The exact output inventory is five regular files totaling **15,401 bytes**: receipt, console, child release, worker attempt and worker profile. Every recorded output hash and byte count matches its saved file; the profile's separate hash/length and console summary also agree.

| Saved artifact | SHA-256 |
| --- | --- |
| `build/private-vascular-streaming-preparation-v1/root-preflight.json` | `dd8b7052250aae4a73c3bc49219732b5e2cb8dc6a85b62e903d025fafaf3c9c0` |
| `build/private-vascular-streaming-preparation-v1/root-supervisor-release.json` | `3a37811aaa2144db98fcc591a95357513652642308a8280afff445eab02c11d9` |
| `build/private-vascular-streaming-profile-v1/receipt.json` | `b9977aed5bfe03d93a794eef977e96ff2d444626541e2c82f625bd7093668805` |
| `build/private-vascular-streaming-profile-v1/worker/profile.json` | `dffc8441d958ac1d0d5d5199ab4ccb05496e1ece1ed8a74c8b65c7192da2fa7b` |
| `build/private-vascular-streaming-profile-v1/worker/attempt.json` | `c2143edafefff46c9070dbc80daba11f75d451a2fe07d02b5f1ff7827b107883` |
| `build/private-vascular-streaming-profile-v1/worker-release.json` | `a42a149a38a4ae11ae91b5c96a16106eede1540ee88b8b7b2e1bcb412400ad6b` |
| `build/private-vascular-streaming-profile-v1/readout-console.txt` | `ad7861b49c5fcf0c133f90a3ffbcb0f57c1273c154807b0d12d75599ea4fad8b` |

The worker attempt remains `reserved_generated_profile` by design; completion is recorded in the bound profile and parent receipt. It is not a second or unfinished worker attempt.

## Time, resources and cleanup

Exactly one child call is recorded (PID 8074), exit 0, no kill reason. Cleanup records direct-child reaping, containment, no remaining group members, no errors and no fallback. The exact command launches the reviewed worker under the prescribed child cache prefix. This audit corroborates the saved cleanup observation; it does not claim continuous process inspection after that observation.

| Measurement | Saved value |
| --- | ---: |
| Contact-kernel profile time, with tracemalloc enabled | 7.883142 s |
| Kernel internal elapsed time | 7.883054 s |
| Supervised child stage | 8.226051 s |
| Recorded parent lifecycle | 8.251027 s |
| Sampled group peak RSS | 43,548,672 B |
| Worker process peak RSS (`ru_maxrss`) | 43,614,208 B = 41.59375 MiB |
| Kernel traced peak allocation | 759,089 B = 0.723924 MiB |
| Kernel traced allocation remaining | 12,125 B |
| Final output | 15,401 B |

These satisfy the unchanged 20-second cooperative kernel budget, 35-second child-stage cap, 45-second accepted lifecycle, 512 MiB sampled group RSS cap and 4 MiB aggregate output cap. Sampled peak output was 13,825 B before final receipt growth; the independently summed final 15,401 B is the correct final total. The worker process RSS peak is slightly greater than the sampled group peak, which is consistent with sampling. No cap was raised.

RSS/output observations can miss short peaks. The recorded lifecycle is taken before the launcher's final evidence checks, although its success path rechecks the 45-second deadline after them. The parent is a cooperating supervisor, not a separate OS watchdog; scientific binary/runtime closure remains unauthenticated. Tracemalloc measures traced kernel allocations and is not total process memory.

## Geometry, work and reporting consistency

Independently recomputed the saved grid/capsule/reference identity hashes and the fixed capsule coordinates using scalar arithmetic, checked the 0.37-radian rotation, 0.7 mm spacing, translation and declared procedural reference coverage, and verified outside-FOV flags. This verifies identity/reporting consistency; it does not repeat the contact geometry computation.

The conceptual reference grid is **256 × 256 × 192 = 12,582,912 cells**. All **3,072** fixed tiles receive coarse consideration for six capsules: **18,432 tile/capsule pairs**. The coarse pass prunes **2,967 tiles (96.582%)**, leaving **105** evaluated tiles. Narrow work totals **1,130,496 cell/capsule pairs**. The largest tile contains **4,096 cells**; both coarse and narrow geometry batches peak at **256**. Reference sampling has **105 calls** and **21,314 distinct touched cells**, equal to the whole-tool union. These counters respect all frozen work limits and do not imply full-volume arrays were allocated.

The repeated action equals the first action exactly. Whole-tool unions are consistent with the per-action and shaft/tip count bounds:

| Contact set | Touched cells | Annotated positive cells | Unknown-coverage cells |
| --- | ---: | ---: | ---: |
| First action | 10,881 | 31 | 5,125 |
| Repeated action | 10,881 | 31 | 5,125 |
| Crossing action | 10,702 | 32 | 4,831 |
| Shaft union | 15,037 | 39 | 7,046 |
| Tip union | 17,778 | 50 | 8,205 |
| Whole-tool union | 21,314 | 60 | 9,956 |

Volumes equal their associated counts times 0.7³ mm³ within floating-point tolerance: whole-tool positive-cell surrogate 20.58 mm³; unknown in-grid cell volume 3,414.908 mm³. Whole-tool and shaft sweeps include outside-FOV uncertainty; tips remain in FOV but include unknown coverage. Coverage is incomplete, the annotated-positive encounter is true, and biological-vessel-free/clinical-injury fields remain null. These are generated annotation contacts, not biological injury or safety outcomes. Removal overlap is not evaluated, and patient/strategy admission remains false.

## Meaning of the feasibility result

This is measured evidence that the reviewed bounded tiled kernel completed this rotated, partially covered, generated large-grid fixture with duplicate and crossing capsule groups using modest process memory. Together with the prior independent tiny geometry controls, it supports continuing toward controlled integration while retaining all strategy/source/admission gates.

The frozen worker also reports **5,624,599** cells for its largest hypothetical legacy bounding box and **539,961,504 bytes (514.947 MiB)** for four coordinate arrays at 96 bytes per cell. Independent scalar arithmetic confirms that estimate. Those arrays were never allocated and the legacy implementation was not run. Consequently there is **no measured baseline runtime, speedup, or baseline RSS reduction**, and comparing the hypothetical arrays directly with traced peak memory is not an apples-to-apples benchmark. The result applies to this one fixture and capsule count; it does not prove performance for arbitrary dense references, real patient loading, complete tool histories, or full matched-method scoring.

## Audit evidence

Executed once from the repository root:

```text
.venv/bin/python -B build/private-vascular-streaming-result-independent-v1/audit_saved.py
```

Exit 0, `passed_saved_metadata_audit`. The auditor refuses subprocess/network operations and patient/model payload suffixes; it uses only bounded regular-file reads, JSON, hashes and small scalar arithmetic.

| Reviewer artifact | SHA-256 |
| --- | --- |
| `audit_saved.py` | `cb02e1a325be4b1e3c0b9d67aa5ec0b2dd745ff81515f556444e2c5cc6e06810` |
| `audit-result.json` | `46ad3df9baf57a80007ca0c97c18a1d4dd4323a25095a5d60aba60584709bce1` |
| `source-and-evidence-snapshot.json` | `bcc0e41d024d1c749aff0bc82086b092670f6ad0913bc991345b9212d80126fd` |
