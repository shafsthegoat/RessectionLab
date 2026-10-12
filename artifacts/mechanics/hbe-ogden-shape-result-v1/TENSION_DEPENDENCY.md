# Remaining tensile-response dependency

Metadata-only follow-up after the completed comparison found no reference-/history-resolved tensile record for HBE_01_03 in the admitted release. Its inventory contains processed tension_c1.csv (1,166 bytes) and tension_c3.csv (1,170 bytes). The README describes moving-average filtering, interpolation and load/unload averaging, with displacement and force columns. It does not supply synchronized time, separated loading/unloading branches, force-zero/preload and reference-gap conventions, or the full processing recipe. The inventoried raw/processed example files belong to HBE_06_07 and cannot substitute for this specimen. This establishes a limitation of the admitted release, not absence of creator-held records.

The missing discriminating measurement is HBE_01_03's synchronized unaveraged force–plate-gap/displacement–time record, including loading/unloading cycles and documented tare, preload, reference gap and processing settings. Without it, the remaining +6.252 mN tensile endpoint bias cannot be attributed uniquely to constitutive error versus reference/history effects. A prospective c1/c3 comparison could test cycle sensitivity but would not recover those missing references. No arbitrary offset, viscoelastic mechanism or patient material property is inferred.

Only existing metadata was inspected; no curve or archive member was reopened, and no new numerical run, source acquisition or contact occurred. Existing specimen and patient roles remain unchanged.

Verified metadata bindings:

- data/mechanics/zenodo-8095559/README.txt: SHA256 `93ded300be0e16ea2cdee8bb2838a149e2da6291d2e84013709dcd3ccebd2ad5`.
- artifacts/mechanics/hbe-8095559-acquisition-v1/metadata-inventory.json: SHA256 `cd71420f5f1da0304fec32651a98eaf44544c696e4d470c753a20fd2434241f5`.

## Saved-curve diagnosis, October 12

A separate read-only calculation on the already exposed comparison narrows the
dependency: a constant vertical force-zero offset alone cannot reconcile the
saved alpha = -8 tensile curve at its fixed displacement reference. At rows 10,
20 and 30 (one-based), the displacement intervals are both 0.249707 mm:

| Quantity | Row 10 | Row 20 | Row 30 |
| --- | ---: | ---: | ---: |
| Displacement (mm) | 0.236596 | 0.486303 | 0.736010 |
| Observed force (mN) | 0.800944 | 2.595043 | 6.744340 |
| Predicted force (mN) | 5.146427 | 9.366177 | 12.995945 |

Observed force increments increase from 1.794099 to 4.149298 mN; predicted
increments decrease from 4.219750 to 3.629768 mN. Their second differences,
`F30 - 2*F20 + F10`, are +2.355199 and -0.589983 mN respectively. A constant
force offset cancels from this calculation. Across all 30 rows, predicted minus
observed force also varies from +0.406217 to +6.796768 mN, rather than remaining
constant.

This is a post hoc shape diagnostic on the same processed, correlated
development observations, not new independent physical validation. It does not
distinguish constitutive-law error from an unknown displacement reference,
prestrain or loading/averaging history. The synchronized reference-resolved
measurement dependency above remains. No offset was fitted, no material law was
changed, and no new solver run or patient-level claim follows from this result.

The measurement specialist's arithmetic was independently recomputed by the
integrator from `execution/comparison.json`, SHA256
`72497329b4671f2569620ac848c019c9f902f6c346d03fc4879500c94284450b`.
Coordinates and observed forces are
`candidates["-8"].fit.modes.tension.{coordinate,observed}`; fitted predictions
are `candidates["-8"].metrics.branches.tension.predicted_at_observed_coordinates`.
Select zero-based indices `[9, 19, 29]` and multiply meters/newtons by 1,000 for
the table units. Equal interval spacing and all 30 stored residuals were checked.
Only saved arrays were read; no new dataset, model fit or substantial allocation
was needed.
