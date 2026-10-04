# Independent PAT05 extraction check

October 4, 2026. A separate imaging/QC agent checked the saved
`artifacts/brain-extraction/PAT05-mps-v1` run and viewed its six native-plane
overlays. The reproducible array audit is
[`brain-extraction-pat05-independent-qc.json`](brain-extraction-pat05-independent-qc.json).
All historical PAT28 artifacts and audit records were left unchanged.

```sh
.venv/bin/python scripts/audit_brain_extraction.py \
  --subject sub-PAT05 \
  --frozen-config artifacts/brain-extraction/PAT05-frozen-experiment.json
```

Creator-source T1 and fractional-annotation hashes match the pinned acquisition
manifest. Its selection record, frozen source snapshot, run implementation,
upstream source/license assets, both model checkpoints, executed MPS runners and
four mask/distance outputs match their retained hashes. The frozen subject,
models, threshold, recorded variant order and five reported execution settings
match the saved run. The declaration timestamp precedes the reported completion
timestamp; these are local records, without external timestamp attestation or
proof that no unrecorded runs occurred. No network inference was repeated by
this audit, and there is only one saved PAT05 run per model variant.

Both masks and both finite predicted-distance fields have the original
160 × 256 × 256 RAS millimeter grid. Maximum physical displacement at all eight
voxel-grid corners is **0.0 mm**. Independently thresholding each distance field
at the recorded 1-mm border, selecting its largest connected component and
filling holes reproduces its saved mask with **zero differing voxels**.
Independent NiBabel orientation reindexing of the source annotation produces
11,437 voxels at the declared intensity threshold of 0.5.

| Independent measurement | Main | No-CSF |
| --- | ---: | ---: |
| Estimated envelope volume, mL | 1402.140 | 1240.898 |
| Connected components | 1 | 1 |
| Input-grid face contacts | 0 | 0 |
| Threshold-derived annotation voxels included | 11,437 | 10,899 |
| Threshold-derived annotation voxels excluded | 0 | 538 |
| Excluded fraction of source annotation | 0% | 4.704% |

The no-CSF omission is explicitly flagged; its excluded voxels span native
indices `[34,133,136]` through `[45,149,150]`. The main model's complete annotation
inclusion does not measure anatomical accuracy. The source annotation is
fractional intensity, not clinical probability, and the threshold remains a
research assumption. No mask was patched or threshold changed in this audit.

The viewed overlays show different model contours around fluid spaces, and the
simple intensity comparator has missing internal regions and inferior support
outside the model envelopes. Those selected planes support an engineering
display check only; they do not establish whole-volume anatomical correctness.
Both estimated envelopes include inferior structures and neither identifies a
reviewed pial surface or cortical access window. The predicted distance fields'
100-mm exterior fill is not measured surgical clearance.

The visual artifact SHA256 is
`3eda847913c76c2c06c9edbcd210c2d79598d49170f2ad0cc9b7f4804065dc02`;
the audited extraction-report SHA256 is
`7be541d3d8f3c2784006aae75cf8e3d420bff37790554eac9eb6990b84abd428`.
All detailed source/model/output hashes are in the JSON audit.

**Review remains required.** This check grants no anatomical approval, cortical
access, functional localization, clinical probability or accuracy claim. PAT05
is a predeclared additional development case, not independent final evaluation.
The auditor's 18 tests include source tampering, physical geometry, mask/distance
consistency and frozen-setting drift. One added regression catches a 0.159-mm
qform/sform corner discrepancy that an elementwise affine tolerance would miss.
