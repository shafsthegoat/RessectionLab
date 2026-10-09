# Independent saved audit: generated 512 × 512 × 64 MR control

**PASS_GENERATED_ARRAY_PATH_ONLY.** The exact authorized generated harness completed once; this review did not rerun conversion, allocate a full array, or open/decode patient data. The observed footprint covers generated imports, data, conversion, serialization, and native round-trip audit. It does not establish a bound for combined acquisition metadata preflight and verified patient loading, or authorize patient conversion.

| Saved evidence | Independently checked result |
| --- | --- |
| Profile | 64 generated 512 × 512 signed16 planes → 512 × 512 × 64 float32 |
| Peak RSS | **387,760,128 bytes / 369.796875 MiB**, below the 512 MiB profile ceiling |
| Recorded worker time | **0.647177 seconds**; this is worker elapsed time, not an independently measured end-to-end parent duration |
| Cleanup | Exit 0, parent reaped worker 98094, no cleanup notes or termination request |
| Output | Five files totaling **67,113,594 bytes**, below 96 MiB; native NIfTI is 67,109,216 bytes including its 352-byte header |
| NIfTI header | 3D float32, millimeter spatial units, qform/sform code 1, unit sample scaling |
| Physical coordinates | Independent header parsing and direct fixture-coordinate checks on all 64 planes reproduce maximum corner error **5.9784513 × 10⁻⁶ mm**; qform/sform corner disagreement is zero |
| Bounded sample check | **3,327 samples across all 64 planes match exactly**, covering corners, center, wrap boundaries, and deterministic dispersed positions |

The sample check read 13,308 sample bytes directly from the saved file and used standard-library scalar math; it did not load NumPy, nibabel, or a full image array. The file was also SHA-256 hashed in 1 MiB chunks. Worker receipts report complete all-sample comparisons before and after serialization; this audit independently repeated a bounded sample check and all-plane geometry checks, not a second full-voxel comparison. All file hashes remained stable across the audit.

Intent, parent, worker, start record, source/test hashes, and the prior independent review bind consistently. The approved harness remained `76e075b096f2aaec73b7ddb1405e088d628138e54777e91229fff2e5d261b596`; source remained `8a6db6cdf6c10f276fd13a4de0d0fe14b62fa5fc368e164d86494cccef47847d`; tests remained `d2c5bb315f70f18336851faf3c108e44e0b4070f6bfdf917493833f409e61289`. Saved startup flags confirm isolated Python, disabled site hooks, and disabled bytecode writes. The generated-only runtime is not the patient pilot's complete runtime/import closure.

One `socket.bind` attempt was recorded as blocked, with zero patient-read attempts. That event is retained as recorded rather than reported as zero. No anatomy, timing availability, registration, preoperative-input, or training admission follows from this generated control. Future patient preparation needs separate metadata-validation and conversion resource accounting and an independently reviewed exact release.

| Bound saved file | SHA-256 |
| --- | --- |
| Parent receipt | `6add4811d86e08d415d4853b82de86b049b3fe472e7cbc1387f8968825709ca6` |
| Worker receipt | `f6c6c9e8a15fdb99731502398e50f65fb4ce3ab75ea12989252908c14c6f08c9` |
| Generated NIfTI | `eafd8ee23a902620ccdb2470b8eaf5608a4d978116c98635cc08300c8e10f8b7` |

The adjacent `audit.json`, `audit_saved.py`, and `review.json` retain reproducible checks, all five output hashes, precise limits, and explicit scope boundaries. The 64 MiB generated image remains in its original ignored output directory and was not copied into this review. Review HEAD: `4b8156c766069d793e1d7c48b241e6e07ae81010`.
