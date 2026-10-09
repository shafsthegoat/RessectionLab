Menichetti one-file sealed byte intake, prepared 2026-10-09

Scope: F_t_curves.mat only, 6,427,434 bytes, publisher SHA256
ce3ee6610977500cc1f999a8b5ab1e620a9551cb0f8d770d3b4e429545c39cb6.
Creator record https://data.mendeley.com/datasets/7sjcn229cw/1, CC BY 4.0.
Metadata and the creator HTTPS redirect-to-S3 HEAD chain are already pinned in
prepared-declaration.json. The future payload request uses that exact S3 URL
without redirects, native verified TLS via Python's default SSL context, and
If-Match with the observed opaque ETag. The ETag is not substituted for source
SHA256. The statistics PDF is explicitly excluded.

The file is eligible for opaque storage only. It includes multiple brains; all
response fields remain sealed until structure inventory authenticates brain
aliases and a donor-grouped response-access role manifest is frozen. No TRAIN,
evaluation, or calibration role is assigned here. Existing HBE, Greiner and
patient roles are unchanged. No MAT decoder is imported by the intake runner.

Transport reuse

run-intake.py is a narrow adaptation of the existing NFBS one-object lifecycle.
It calls the unchanged acquire_public_case.acquire_file and unchanged SHA256
verify_file. Frozen source copies and their pins are in the declaration. The
acquire_btc_case module contributes only RejectRedirects; its MD5 verifier is
not substituted. real_intake_io supplies immutable publication and termination
cleanup. The source helper performs exact length/SHA256 verification before
atomic publication. The wrapper rejects an ignored Range before the helper can
restart/truncate a partial, and holds both queue locks throughout transfer.

At most three durable transport intents across restarts. Valid short transfers,
timeouts, connection/DNS failures and HTTP408/429/500/502/503/504 may retry.
Backoff is 5 then 15 seconds;429/503 wait at least60seconds and respect longer
Retry-After. TLS, source/ETag, SHA256, oversize, Range or scope failures stop
without automatic retry. A missing result for a prior intent requires review.
Require100GiBfree reserve after the remaining file bytes. No payload request
has been made. Payload-manifest.json is directly readable by the existing
helper's dry-run command, but its unguarded download CLI is not the launch path.

Queue

Predecessor: ReMIND SEG run20261009T090019816874Z-e7318361, released declaration
da9401bd5cb5a6936c389a401ea41ae44bc68734d3765268e25ece038452dd0c.
The coordinator reports242/242 complete. Bind its canonical completion receipt
SHA256 after verifying it. This prepared declaration remains false-release;
after root review, save a separate released declaration under
data/acquisition/menichetti-mechanics-v1/declaration.json with the same scope and
helper hashes, the exact predecessor hash, execution_released=true, and
scope_frozen_before_payload_access=true. Record its final SHA256 before launch.
Do not launch this command before that root-reviewed release exists:

.venv/bin/python build/menichetti-intake-preparation-v1/run-intake.py \
  --declaration data/acquisition/menichetti-mechanics-v1/declaration.json \
  --declaration-sha256 RELEASED_DECLARATION_SHA256

The present safe preparation check is:

.venv/bin/python build/menichetti-intake-preparation-v1/run-intake.py \
  --declaration build/menichetti-intake-preparation-v1/prepared-declaration.json \
  --declaration-sha256 PREPARED_DECLARATION_SHA256 --check-only

Use the actual hash from SHA256SUMS; placeholders are deliberately not runnable.
The coordination agent owns queue ordering, and this artifact does not start a
second scheduler or automatic acquisition framework.

Narrow post-intake structure proposal, not executed or released

1. Verify the successful byte receipt and exact original SHA256. Read only the
MAT format/header to select a format-aware metadata route; no loadmat call.
2. For MATv5 use scipy.io.whosmat for top-level variable names/shapes/classes.
This cannot by itself establish nested region/trial mappings. If nested facts
are needed, separately review a bounded structural-tag walk that reads names,
flags and dimension tags and skips numeric payload tags without interpreting
them. Compressed containers may require opaque decompression to reach nested
metadata; explicitly log that byte exposure and retain no numeric arrays.
For MATv7.3 use HDF5 object/dataset metadata only (names, shapes, dtypes and
attribute names); do not read dataset or attribute values. Neither route should silently fall
back to full scientific deserialization if the layout is unsupported.
3. Emit only brain aliases, region keys, field names, dimensions/classes,
number of trial columns, and presence/names/shapes of time/position/command
channels. Treat a time-like name as a candidate, not proof of unit or alignment.
Do not output minima, maxima, means, samples, plots, nonfinite counts or force
derived quality flags. Do not open numeric time channels in this first stage.
4. Inventory all donors structurally before freezing whole-donor roles. Bind
each role to original hash and explicit structural path; group all regions,
sites and repeated tests from one brain. Unknown alias/donor identity remains
an explicit dependency; do not reclassify a specimen as an independent donor.
5. Qualify timing/contact/channel semantics from creator evidence before any
force-response access. If no timing fields exist, preserve that negative
finding. No response-dependent contact alignment or unpublished parameter
statistics may be used to silently turn this into unconditional force truth.

Stage limits: one small file, no response interpretation, no mechanical fit,
no solver, no patient data, no changes to existing sealed split contracts.
