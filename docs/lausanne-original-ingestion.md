# Lausanne original-image ingestion

The first acquired pair is `sub-000/ses-20110101`, prospectively TRAIN. It comes
from [ds003949 v1.0.1](https://openneuro.org/datasets/ds003949/versions/1.0.1),
Git `896b8846d899acee68c0246cc987ca96e77267d4`, under the source's CC0 release.
Cite Di Noto et al., [Neuroinformatics, 2022](https://doi.org/10.1007/s12021-022-09597-0).
Only provider-described original T1/TOF images and source JSONs were acquired;
no skull-stripped, N4, atlas-registration or learned-model derivative was used.

The metadata-only split was frozen before image access:

| Role | Unique people | Sessions |
|---|---:|---:|
| TRAIN | 199 | 210 |
| SELECT | 43 | 43 |
| MEASUREMENT_EVAL | 42 | 43 |

The [cohort manifest](../manifests/lausanne-component-cohort-v1.json) groups all
sessions of each person. Its fixed identity hash order is stratified by the
published control/patient field; 000 is the explicitly declared TRAIN pilot.
This is one-site component development, not a held-out-site clinical study.
Twenty people also occur in TopCoW. Those derivatives inherit 17 TRAIN,
one SELECT and two MEASUREMENT_EVAL assignments; they add zero unique people.
All earlier BTC, UPenn, UTSW and other patient roles remain unchanged.

## Reproduction and current result

```sh
.venv/bin/python scripts/acquire_lausanne_pilot.py prepare
.venv/bin/python scripts/acquire_lausanne_pilot.py acquire
.venv/bin/python scripts/audit_lausanne_pilot.py
.venv/bin/python scripts/inspect_lausanne_pilot.py
.venv/bin/python -m pytest tests/test_lausanne_original_pilot.py -q
```

Completed acquisition/audit/display receipts are immutable; their commands
refuse to overwrite existing results. Downloads use resumable partials and
verify fixed S3 versions, source-annex sizes/MD5 and sidecar SHA-256 before
publication. `prepare` is idempotent for an identical declaration. Originals
and the figure stay in ignored local `data/` and `outputs/`; no patient payload
is committed. The full eligible TRAIN intake is still outstanding.

The [result](../artifacts/lausanne-original-pilot-v1/RESULT.md) links compact
acquisition, independent byte/geometry and display receipts. The first run
acquired 36,661,729 bytes in 157.357 seconds. Two source/control checks pass;
these are additional to the separately committed 42 policy checks.

## Fidelity limits exposed by actual data

- T1 has 37 × 420 × 448 samples with approximately 3.9 × 0.558 × 0.558 mm
  spacing. Its sparse direction must not be treated as isotropic fine detail.
- TOF has 350 × 448 × 160 samples at 0.469 × 0.469 × 0.7 mm. Central-plane
  inspection shows a regional slab; full-head vascular coverage is unverified.
- T1 qform/sform agree within 0.000422 mm over image corners. TOF has only an
  active qform. Both have finite values and valid numerical header geometry.
- The TOF sidecar's DICOM directions differ by about 1.70° from its axis-aligned
  NIfTI directions, even allowing storage axis permutation/sign changes. The
  sidecar's multislab spacing metadata also differs from voxel spacing. Retain
  both originals and seek source interpretation; do not repair them by assumption.
- Scanner-frame provenance and T1/TOF registration remain unresolved. Header
  validity, one shared subject ID and visually plausible images cannot establish
  cross-scan alignment or clinical anatomy accuracy.

These scans supply **zero recorded surgical transitions and zero optimizer
updates**. They are component inputs, not a validated surgical planning case.

## Annotation and expansion queue

[TopCoW's external release](https://zenodo.org/records/15692630) contains actual
CoW voxel labels mapped back to original Lausanne frames. Metadata-only ZIP
inspection identified 17 TRAIN masks totaling 2,665,580 extracted bytes. No mask
payload has been acquired. Original MRI CC0 does not establish the annotation
license: the release API has no explicit license, and challenge terms' scope
for these external annotations is unresolved. No new agreement was accepted or
organizer contacted.

[Methods sections 2.2/S3/S6](https://arxiv.org/html/2312.17670v5) describe a mix of
manual labels and model-assisted labels followed by human correction. Exact
Lausanne per-case provenance and initializer ancestry remain unspecified. Human
verification does not by itself settle that lineage. Annotation rights,
lineage, image identity, geometry and coverage must pass before training use.
CoW labels would still cover selected main vessels, not all cerebral vasculature.

The full source-indexed intake is implemented below. Record each failed or
deferred record; unresolved scanner-world frames cannot enter spatial planning.
At the observed pilot throughput, the roughly 10 GB TRAIN intake could take many
hours; use measured batches and checkpointed receipts, not an unbounded download.
Separately resolve human annotation rights/lineage or acquire an alternative
eligible label source. No new RL sweep is justified by this ingestion result.

## Full TRAIN source index and bounded intake

The [full TRAIN index](../manifests/lausanne-train-originals-v1.json) now resolves
all **199 people / 210 sessions / 840 source files**, totaling **10,020,802,851
bytes**. Its SHA256 is
`6f1fc7812af0d66550076aa08d37d7f36f08d764fdab701bbbc9ca0608629e66`.
Every image is bound to the pinned Git-annex size/MD5 and a matching immutable
S3 object version; every source JSON is retained verbatim in the index.
The first index attempt retained 201 sessions and nine actual network failures;
one metadata-only resume resolved all nine. [Failure and index receipts](../artifacts/lausanne-train-intake-v1/source-index.json)
preserve this history. Index completion is not image acquisition or training.

The new `scripts/lausanne_train_intake.py` uses one supervised acquisition worker
at a time. Existing originals are verified against source fixity, and every
cached QC receipt resolves to a retained source/runtime record and code snapshot.
Sessions remain distinct from people. The full denominator includes failed,
unattempted and deferred records; failures cannot disappear through resumption.
The byte budget counts full source sizes per attempted session, not measured
network traffic. Each image decode is limited to 512 MiB of float32 samples;
this is not an operating-system memory cap. A worker has a separate watchdog.
Ordinary parent termination runs cleanup and closes receipts; SIGKILL or an
unwritable filesystem cannot guarantee a final receipt.

Use `--session sub-000/ses-20110101 --existing-only` for an acquisition-free
recheck of the existing pilot. The explicit identity restriction applies even
when that session already has a cached receipt. SELECT and MEASUREMENT_EVAL
payloads remain outside this runner's input allowlist.

The [bounded acceptance runs](../artifacts/lausanne-train-intake-v1/RESULT.md)
completed one fresh existing-pilot QC worker and one cache-only verification;
each excluded the other 209 sessions. Fourteen focused software controls pass.
Reproduction after the original pilot has been acquired:

```sh
.venv/bin/python scripts/lausanne_train_intake.py index
.venv/bin/python scripts/lausanne_train_intake.py batch \
  --session sub-000/ses-20110101 --existing-only \
  --index-sha 6f1fc7812af0d66550076aa08d37d7f36f08d764fdab701bbbc9ca0608629e66 \
  --max-seconds 120 --max-bytes 36661729
.venv/bin/python -m pytest tests/test_real_intake_io.py -q
```

Subsequent bounded TRAIN acquisition uses the same command without `--session`
and `--existing-only`, with explicit time/source-byte budgets. Retain the lock,
frozen index, source snapshots and attempted-run receipts between batches.

A [three-request transport diagnostic](../artifacts/lausanne-train-intake-v1/transport-diagnostic.json)
compared urllib, curl and urllib again on the same existing first MiB. The
repeat urllib measurement was close to curl (9.046 versus 8.875 seconds).
Retain the existing downloader; this small ordered sample does not justify a
client replacement or establish full-cohort throughput.

## Scanner-frame provenance follow-up

Read-only upstream investigation found the pilot's pointer/sidecar mismatch
already in both v1.0.0 and v1.0.1. Release changes do not explain a geometry
correction. The NIfTI `descrip` fields contain `6.0.1`; this is consistent with
an FSL version but does not prove which transformation occurred. The TOF has no
active alternative sform or extension carrying the missing transform, and its
JSON lacks DICOM position information. Direction metadata alone cannot recover
a scanner transform.

The [author's processing response](https://github.com/connectomicslab/Aneurysm_Detection/issues/10#issuecomment-3538848996)
describes BET and unavailable original N4/registration scripts. A released
193-byte `out_T1_2_TOF_0GenericAffine.mat` and registration-quality metrics are
available, but are algorithmic estimates, not measured landmark errors. The
filename does not establish transformation direction or ITK point/image
conventions. No transform was applied. Source-frame provenance and independent
same-person registration validation remain required for spatial use.
