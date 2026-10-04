# Released saved-query diagnostic

The first and only execution completed successfully from immutable commit
`b0f5635e8f77a7d223b25fecb3c47435f372d3fa`. The release receipt was written before
execution. All 16 copied source/input files matched before and after the run,
with no added files in the isolated source tree. No live project code was
imported. No capsule-cover computation, patient loading, simulation or training
ran.

The saved trace reconstructs **22,364 ordered queries and 8,812 distinct argument
keys**, within the bound invariant source frame. The first phase therefore has
13,552 repeated accesses. Of the distinct keys, 1,368 occur once, 1,336 twice and
6,108 three times. The four inventory query counts are 8,812 / 7,444 / 6,108 / 0.

The complete key set is smaller than the original 16,384-entry cap. Raising that
entry cap alone cannot help this fixed trace. With unlimited payload, the
entry-only model predicts 13,552 first-pass hits and 22,364 repeat-pass hits at
that entry count. Those hypothetical counts do not predict the byte-limited
cache's timing. Entry-only upper bounds require the same no-bypass admission
policy; the original experiment recorded zero bypasses.

Full cover-array sizes and byte-weighted reuse distances remain unknown. Active
contacts were filtered by tissue and shaft coverage arrays were not saved. This
result therefore does not select a 64 MiB, 128 MiB or other byte capacity. A
separate bounded pure-geometry diagnostic would be needed to measure those
sizes before declaring another capacity experiment.

External child wall time was **1.108213417 s**, including interpreter startup and
exit. The producer's **1.022518917 s** field ends before writing its final summary
and status. Archive preparation and post-run hash checks are outside both
measurements. The child passed its cooperative 60-second and 512 MiB process
checks; the external launcher also enforced a 60-second timeout. **Exact peak RSS
was not retained**, so no peak value is inferred from the limit. No rerun was
performed to obtain a cosmetic memory measurement.

The compressed ordered argument artifact is **1,197,360 bytes**, SHA-256
`534759de882aa54cde6170a689f35d18f61a15e3ce96b690cc9fe56f10517cb5`.
Raw diagnostic output is under `artifacts/diagnostics/native-cache-query-keys-v1`.
The external execution receipt records its four original output hashes.
`root-release.json` binds the exact archive, complete 16-file copy list, command,
declaration, source and both review receipts. The minimal archive SHA-256 is
`ed3d7e4e5a9208827cc008d17f7110d7a8be6b3bb886642e9d5811d3b73c66be`.

Independent saved-output verification passed on its first attempt. It checked
all 22,364 reconstructed arguments in order, all 66 certificates and 11,182
microsteps, the complete reuse histograms and LRU results, the output authorities
and all 16 archived files. The receipt is
`artifacts/native-cache-query-result-audit-v1/audit-01.json`, SHA-256
`cc716d7b12a54a4c3f899b974b23ee4bd520f8ea8947d57320241d3d29128797`.
