# Remaining tensile-response dependency

Metadata-only follow-up after the completed comparison found no reference-/history-resolved tensile record for HBE_01_03 in the admitted release. Its inventory contains processed tension_c1.csv (1,166 bytes) and tension_c3.csv (1,170 bytes). The README describes moving-average filtering, interpolation and load/unload averaging, with displacement and force columns. It does not supply synchronized time, separated loading/unloading branches, force-zero/preload and reference-gap conventions, or the full processing recipe. The inventoried raw/processed example files belong to HBE_06_07 and cannot substitute for this specimen. This establishes a limitation of the admitted release, not absence of creator-held records.

The missing discriminating measurement is HBE_01_03's synchronized unaveraged force–plate-gap/displacement–time record, including loading/unloading cycles and documented tare, preload, reference gap and processing settings. Without it, the remaining +6.252 mN tensile endpoint bias cannot be attributed uniquely to constitutive error versus reference/history effects. A prospective c1/c3 comparison could test cycle sensitivity but would not recover those missing references. No arbitrary offset, viscoelastic mechanism or patient material property is inferred.

Only existing metadata was inspected; no curve or archive member was reopened, and no new numerical run, source acquisition or contact occurred. Existing specimen and patient roles remain unchanged.

Verified metadata bindings:

- data/mechanics/zenodo-8095559/README.txt: SHA256 `93ded300be0e16ea2cdee8bb2838a149e2da6291d2e84013709dcd3ccebd2ad5`.
- artifacts/mechanics/hbe-8095559-acquisition-v1/metadata-inventory.json: SHA256 `cd71420f5f1da0304fec32651a98eaf44544c696e4d470c753a20fd2434241f5`.
