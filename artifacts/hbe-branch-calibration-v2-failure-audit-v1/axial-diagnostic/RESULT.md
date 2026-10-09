# Released axial CSV framing failure

V2 failed because its explicit schemas declare `header: null`, but both released axial members begin with exactly `displacement,force\n`. The existing parser already supports exact headers: `mechanics_hbe_access.py:427–429` removes a first row only when it equals the declared list. With `null`, line433 attempts `float("displacement")` and raises. No parser modification is needed.

| Allowed member | Bytes | SHA-256 | Framing |
|---|---:|---|---|
| compression_c3.csv | 1,222 | bada734592217ce8134317cbe929223be0436e2d6010a7df4f249b6678ba3a43 | Header +30 records; all2 columns |
| tension_c3.csv | 1,170 | 750bd806e5fc18083e157a31cb019456bd80ef779cf97d5f1d298e52c9d96fc8 | Header +30 records; all2 columns |

Both use strict UTF-8 without BOM, comma separation, LF line endings, and no blank records. This diagnostic read each of those two members once after authenticating the original archive SHA/size and each member's size/CRC. It retained only headers, counts and identities; no numeric response values were converted, reported or fitted. Thirty following records is a framing observation, not validation that all numeric rows will pass the parser's later checks. No torsion member content was opened.

## Preserve the access history

The v2 state remains `failed_or_incomplete`, with calibration access attempted=true, responses accessed=null, held-out attempted/accessed=false and native/mesher calls=0. Do **not** rewrite null as false: the released reader had already loaded a member payload. The ledger contains one `calibration_attempt` listing both allowed members and no `calibration_completed`. That event records intended scope, not a per-member success record. The committed loop order and traceback show failure while parsing the first (compression) member; tension was not reached by that original loop. Both were subsequently read under this separate root-authorized framing diagnostic, which is recorded here. No fit, prediction freeze, or numerical-confirmation output was produced by the attempt.

## Minimal separate v3

Create a v3 study/source archive/release/output root, preserving v1 and v2 files, started markers, ledgers and failures. Change only the two axial `header` fields to `["displacement", "force"]`; keep delimiter, columns, units, signs, original member split, fit/objective, physical model, meshes/schedules, native backend/runtime, all tolerances and all stage/aggregate/output budgets unchanged. Version the three branch modules by mechanical identity/import/schema edits and add a validator for the pinned v2 header failure and access ledger. Retain the v1 runtime-failure lineage. The v3 validator must allow exactly the recorded attempted=true/responses=null calibration state, require zero calls/no fit/no completed calibration or freeze, and reject any held-out access. A fresh root release must bind the new header metadata; the old release cannot authorize it.

Use the unchanged exact-header parser—no autodetection, generic first-row skipping, field renaming, sign inference, dropped rows or tolerance adjustment. Focused controls should exercise the exact header, mismatches/absence, unchanged30-row handling using analytical values, all non-header fields unchanged, predecessor/access-ledger joins and sealed held-out access. This diagnosis creates no v3 source, manifest or release and performs no rerun.

**Remaining scope limit:** both torsion headers remain uninspected, and their v2 schemas still say null. Axial evidence does not establish torsion framing, so v3 completion cannot be promised. Do not guess or silently inspect them. Root should resolve whether to authorize a separately recorded header-only framing check before the expensive new attempt; absent that authorization, retain the existing torsion schemas and their explicit unresolved risk. Such a check must not use held-out responses to alter the scientific model or fit.

Original archive, all reviewed source/receipt bytes and the failed v2 records were rehashed unchanged. `diagnosis.json`, the prior-written inspection declaration, and the reproducible inspection helper retain exact scope/provenance.
