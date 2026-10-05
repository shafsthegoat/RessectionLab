# Four real BTC structural bundles

All four authorized preparations completed on October 4, 2026 in 21.020504 seconds. Each case ran once and exited successfully; no failure, retry, replacement, training, inference, registration, or download occurred. The saved bundles reopened with identical semantic and planning identities.

Execution used Git archive `086b7cda8d44733355a926a5990694aba0f2f137`, with 131 frozen files covering source, scripts, manifests, and the acquisition receipts. Python ran with `-I -S` and explicit snapshot source/script directories plus the existing virtual-environment dependencies. Runtime receipts verify that every imported `resectionlab` module came from the frozen snapshot, including core and imaging. All frozen files and all 16 unique retained source files matched their initial SHA-256 after preparation. Maximum child process RSS was 430,014,464 bytes on macOS, with one numerical thread requested.

| Case | Frozen development role | Target voxels at 0.5 | Bundle bytes |
| --- | --- | ---: | ---: |
| PAT22 | Population training | 13,915 | 9,299,994 |
| PAT25 | Population training | 16,526 | 9,890,058 |
| PAT26 | Checkpoint selection | 55,312 | 9,722,875 |
| PAT27 | Checkpoint selection | 11,983 | 9,168,261 |

Bundles are retained outside Git at `outputs/cases/BTC-sub-PAT##.ressectionlab`. The corresponding preparation reports retain native affine matrices, source-manifest hashes, declared annotation threshold and sensitivity counts, derived volumes, source identities, unknown availability times, and role restrictions. The batch receipt retains each bundle's SHA-256, semantic/planning identities, size, log hash, report hash, and runtime hash. Source data remain untouched in `data/diffusion_source/ds001226-v5.0.1`.

| Receipt | SHA-256 |
| --- | --- |
| `root-release.json` | `62fcd60d8dde099ee73f10ab1ede01e0a12b6a07e1c442aa08ded75b373942de` |
| `batch-record.json` | `a92ad03376373e12f3a23bc2cb002e3dbe47fa0c1a1a56ef0dd2f6b214aaed49` |
| Frozen Git archive, retained under ignored `build/btc-spatial-preparation-086b7cd/source.tar` | `a1ec1a1538fcf047ddefe00c25445aa80aeb90787caffce0bce8dd8768139441` |

These are real public patient scans with supplied annotation-derived reference targets. MRI, affine, and threshold-derived binary masks are embedded; original fractional annotation values remain in their unchanged external source files, referenced by SHA-256. The annotation is not calibrated probability, an observed removal, or verified anatomical truth. The bundles explicitly mark targets as inspection/hidden-environment reference only and prohibit their use as scan-only actor inputs. Checkpoint-selection cases prohibit policy training. These metadata contracts still require enforcement by downstream consumers.

Every bundle retains `brain_mask=None`, full-head coverage, forbidden nonzero-MRI access support, unavailable directional diffusion, pending visual annotation review, and blocked automatic cortical access. No support proposal or functional localization was attached. Unknown-timed pathology/context remains excluded from planning inputs. PAT29 and PAT31 remain absent and unopened, and no final-cohort assignment changed. Preparation and persistence checks establish engineering integrity; independent anatomical review and planning eligibility remain outstanding.
