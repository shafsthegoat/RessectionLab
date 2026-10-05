# Fixed support proposals saved with real BTC scans

Four separate proposal-enriched bundles were saved and reopened in 23.764932 seconds using frozen source from `ea501a8991e9a4ec6a6154f117c48a8b6ff1e390`. Each contains one estimated, `review_required` main-model envelope. Every working `brain_mask` remains absent; no review, cortex localization, or cortical-access permission was added. PAT22/PAT25 retain training roles, and PAT26/PAT27 retain checkpoint-selection roles with policy training prohibited. Source targets remain excluded from scan-only actor input.

All four inference outputs were fixed before this stage opened target annotations. No inference or model training was repeated. The original MRI, affine, active/source annotation masks, source references, context, metadata and planning hashes are unchanged. Adding the proposal changes semantic identity and advances revision 2 to 3. Saved/reopened semantic and planning identities match exactly. All 59 retained source/model/inference files and 91 frozen implementation files remained unchanged; maximum integration-process RSS was 599,097,344 bytes.

| Subject | Annotation voxels | Voxels outside fixed main envelope | Envelope volume, mL |
| --- | ---: | ---: | ---: |
| PAT22 | 13,915 | 0 | 1,458.746 |
| PAT25 | 16,526 | 0 | 1,506.341 |
| PAT26 | 55,312 | 0 | 1,767.171 |
| PAT27 | 11,983 | 0 | 1,794.076 |

Each mask has one connected component and zero input-grid face contacts. Mask and predicted-distance grids have zero measured corner displacement from their source T1 affines; predicted distances are finite. These are engineering checks. Complete inclusion of a supplied threshold-derived annotation is not anatomical accuracy, clinical benefit, or a reason to accept or select a model. No target voxels were unioned into an envelope and no settings changed after overlap was measured.

The outputs are `outputs/cases/BTC-sub-PAT##-structural-evidence.ressectionlab`, for the four declared IDs only. Original case bundles remain unchanged. Post-inference QC reports are separate derivatives at `outputs/brain-extraction/BTC-spatial-main-v1-qc/sub-PAT##/brain_extraction_report.json`; original inference reports remain unchanged. Each portable bundle embeds the exact new report text/hash and proposal mask. Predicted distance arrays remain external, with source hashes; their 100-mm exterior fill is not calibrated clearance. Independent reconstruction, native-plane visual review, and bundle reopening are recorded separately by the imaging reviewer. The creation receipt preserves its historical `independent_QC_pending` status.

An initial setup attempt failed before patient arrays were opened: a receipt helper used system Python without `hashlib.file_digest`, leaving the release file absent. The integration driver stopped on that missing file. `attempt-01-failure.json`, the original driver and its log retain this failure. Attempt 2 used the declared app interpreter and a separate destination; it also corrected a statically detected, previously unreached evidence-property typo. No production module, original source, or model output changed.

`attempt-02/integration-driver.py` is the exact successful driver. `attempt-02/root-release.json` records the command and source binding; `execution.log` retains all four outcomes. `integration-record.json` binds every original and derived identity, full QC metadata, roles, import origins, and save/reopen checks. Its SHA-256 is `548681883bfaeceb97176a59497be9d1c975b46d634f2f37d8b2357d6c4ee68e`. Raw medical arrays and derived bundles remain outside Git. Neither this persistence check nor later engineering review grants anatomical approval or establishes clinical safety.
