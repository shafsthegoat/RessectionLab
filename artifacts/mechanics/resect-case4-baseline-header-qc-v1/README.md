# RESECT Case4 baseline header audit

Completed under root's baseline-only release after access contract `cb9f71b`
and acquisition `65bf883`. The original compressed T1, FLAIR and before-US bytes
match the acquisition receipt's SHA256, provider MD5 and size. Only 348
decompressed NIfTI header bytes per image were requested/interpreted. No image
array, during-US file or landmark file was opened. No source was modified.

| Original | Native dimensions | Header spacing (mm) | Voxel-axis codes | Active transform |
| --- | --- | --- | --- | --- |
| T1 | 256 × 256 × 192 | 1 × 1 × 1 | P, S, L | sform code 1 |
| FLAIR | 256 × 256 × 192 | 1 × 1 × 1 | P, I, R | sform code 1 |
| Before-US | 337 × 303 × 293 | 0.213 × 0.213 × 0.213 | L, I, P | sform code 1 |

All are float32, millimetre spatial units, finite nonsingular sforms, slope 1,
intercept 0. All qform codes are zero: unused zero quaternion/offset fields
provide no alternative geometry and are not a disagreement requiring repair.
No header normalization or qform substitution was applied.

**Frame findings.** T1 and FLAIR have different affines despite equal dimensions.
The header-derived index mapping includes two axis reversals and an oblique
offset; copying voxel indices or treating a simple array flip as registration
would be incorrect. Their overlapping physical extents are not proof of aligned
anatomy or proof that the published upstream T1-to-FLAIR registration is present
in this exact release. The before-US grid is separately oblique. Preserve each
original sform and transform physical points, not array indices.

Normalized-column Gram deviations are 2.17e-9 (T1), 3.22e-9 (FLAIR) and 7.04e-7
(before-US). These small nonorthogonalities are recorded, not silently removed
or attributed to a verified acquisition mechanism. General-affine physical
coordinate handling avoids needing a header alteration. Axis codes describe the
NIfTI declaration; independent anatomy/landmark frame acceptance remains false.

**Required next work only:** after the separate measurement-access release,
use the audited source-only parser to freeze the six-B/V partition and inspect
the permitted MRI-before baseline pairs. Audit point-to-image physical bounds
and pair direction before claiming a canonical landmark frame. A later bounded
baseline-image review must check T1/FLAIR alignment and before-US overlays;
the permitted baseline rigid fit and residual/QC record must remain separate
from during-US validation. A brain envelope still requires a declared extraction
and unreviewed geometry QC; full-head/nonzero image support is not a brain mask.
This receipt authorizes no segmentation, registration, patient model or V reveal.

`probe.py` is the exact executed source, SHA256
`4097493f70166b073f20c923c0b035e05210ad71c875e333c4cb43396a35ba30`.
`header-receipt.json` SHA256
`18d6e6306e38f409a314ba180d7d0a21e9a7ee02677f34a1d1fd28ea1294d692`
contains the exact affines, raw-header hashes, source hashes and runtime paths.
The bounded process completed in **0.220 s**, with **42.4 MiB** child lifetime
peak RSS and **63.9 MiB** maximum sampled combined parent/child RSS, below the
55-second/1-GiB limits. RSS polling is not continuous enforcement; its limitation
is retained in `resource-receipt.json`. The initial dependency probe found no
`psutil`; no package was installed and system `ps` supplied the monitor.

The empty `probe.log`, release and resource receipts are retained. The script
refuses existing receipts; no patient-data retry occurred. Gzip may buffer
compressed input beyond the requested header, but voxel values were never
decompressed into or interpreted as image arrays.
