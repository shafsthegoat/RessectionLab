# Independent ReMIND cohort-role and raw-source preflight

Decision: **GO for freezing the source-only patient roles; payload release remains
separate.** No new patient image or label body was opened in this review. The
reviewed proposal is `build/remind-cohort-expansion-preparation-v1/proposed-cohort.json`,
SHA-256 `75aa5ae21e7af5c55ae6a610f248138ca29310e5fa59ab75c683a955d587fd4f`.
If root changes its declaration status or freeze timestamp, all downstream
manifests that bind this SHA must be rebound to the **committed frozen** hash
before any GET.

## Patient roles and eligibility

- Independently recomputed `SHA256("RessectionLab:ReMIND:component:v1|" + exact
  PatientID)` for every one of 113 newly assignable source people. Sorting by
  digest then PatientID reproduces every stored rank and role: 79 TRAIN,
  17 SELECT, 17 MEASUREMENT_EVAL. The previously exposed `ReMIND-001` remains
  DEVELOPMENT with its original `development_annotation_assisted_geometry`
  role; the existing declaration hash
  `3dad212f49655937d8ef3d0dff998eb796de9f885c4328e9f4202503157ac08f`
  matches the source. All source timepoints, modalities, annotations and
  ReMIND2Reg derivatives must inherit the same person role. SELECT/EVAL have
  no payload in this scope, and later QC failure cannot cause reassignment.
- Official clinical workbook SHA-256
  `5c416ec4a4aa40247feee993af8af017e73ec3455fd05bab8c30ee03627343e8`
  and 118,221-byte size match the existing case-001 source binding. The
  restricted metadata receipt SHA-256
  `425d93e234e1354830c45b95cde68854668044f258bce3a81c2dafe1b9d87694`
  declares only `Case Number` and `Histopathology` cells selected. Recomputed
  exact-source-label eligibility: 88 people have Astrocytoma, Glioblastoma or
  Oligodendroglioma (DEVELOPMENT 1, TRAIN 63, SELECT 13, EVAL 11). The other
  26 are **outside this strict predicate**, not necessarily nonglioma; raw
  labels such as low-grade glioma remain unchanged. The published broad 92
  count is a different definition, not a license to reclassify cases.
- All 79 TRAIN patients may be candidates for **generic MR/US imaging component
  intake**. Only the 63 strict-label TRAIN patients are candidates for a later
  separately admitted glioma-specific policy study, after anatomy, frame and
  timing QC. No patient is currently marked glioma-policy admitted. Diagnosis
  availability before surgery is unknown; it is cohort metadata only and must
  not enter actor, planner or reward inputs. No outcome, image quality or model
  score chose a role. This is an internal ReMIND split, not independent-site or
  clinical validation; cross-collection identity and pretrained exposure remain
  unresolved.

## Exact source bounds and checksum paths

- The IDC v24 index hash is
  `94ff95473e68843682c1fa1a7889daca178b4fc5280ef839c4acd0b0535fa571`;
  projected nonclinical series metadata SHA-256 is
  `8301d9890e97316c8905448c9ec78e851d43ed18b383455ae138f4b13a8023be`.
  The proposal and `train-raw-series-index.json` (SHA-256
  `4a809bf729f621f30dbb988b257eb107ae380b3d085633e0c655f679da4fed4a`)
  list 692 unique MR/US series belonging **only** to the 79 TRAIN people,
  with 59,520 indexed objects. SEG series remain metadata-only and are not
  silently mixed with raw observations.
- Exact S3 prefix listings bind 59,520 unique object keys and
  **30,173,493,690 bytes** in `train-exact-object-manifest.json`, SHA-256
  `1cbd5a5ec5a53c0f06066876f0f2c031d610b96ed84db21cd96fa4931d3f4d23`.
  These exact object lengths supersede the coincidentally equal index sum of
  rounded `series_size_MB` values. Every listed object belongs to TRAIN; no
  protected payload or image body was accessed while listing. For 59,297
  single-part objects, the source ETag is pinned as an expected MD5. The
  remaining 223 multipart ultrasound objects have no MD5 inference from S3
  ETag; official GCS HEAD metadata binds each matching key and size to a
  generation and whole-object CRC32C in `train-us-checksum-manifest.json`,
  SHA-256 `2fcf0fff4bf55dfcee2ff64684473045d4d2fe089f203b75f132a23caa36fa00`.
  These are **pre-download source expectations**, not completed payload fixity.
- Continue with existing verified-TLS, redirect-rejecting, resumable transport.
  Check exact URL/key, length, source identity, and MD5 for single-part objects;
  use pinned generation and whole-object CRC32C for multipart GCS objects.
  Record local SHA-256 and mark every downloaded series unreviewed until its
  intended geometry/anatomy/timing QC passes. Do not publish or train from an
  object merely because its metadata or transfer checksum passed.

This review does not test image content, clinical eligibility beyond the narrow
source-label stratum, registration, surgical actions, forces, RL transfer or
patient outcome prediction.
