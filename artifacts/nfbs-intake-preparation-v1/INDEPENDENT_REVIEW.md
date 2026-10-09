# Independent NFBS acquisition-only preflight

Decision: **GO for a bounded original-archive byte transfer after root freezes
the source-family role and issues a hash-bound execution release. Current
prepared declaration is correctly HOLD (`execution_released=false`).** This
review made no archive GET, image/label read, extraction, QC or training call.

## Source and scope

- Proposal `build/nfbs-intake-preparation-v1/proposed-cohort.json` SHA-256
  `6699327efb00f585e2727ab61a9420efda2a1f3c94022112178c8c24c2b838ad`;
  prepared declaration SHA-256
  `c1368bd185a6cde6aa22bc76937f9b4ad5be875c9e4c129a41f5a55a1a47bd73`;
  frozen runner source SHA-256
  `69e40965e875f33b92b98fc9050c87da574466ee54906b5e013376c3e8009966`.
  The runner's zero-network `--check-only` command passed all bound file hashes
  and printed `prepared_contract_valid`; it cannot start a payload request while
  `execution_released` remains false.
- Compared with the retained initial proposal, the only substantive change is
  semantic: two obsolete synthetic-free/prohibited-ancestry flags became explicit
  ancestry preservation and permission for later synthetic/simulator learning.
  The linked source-audit hash and cohort bindings were updated accordingly.
  Payload, roles, transport, resource bounds, and no-current-label-admission
  controls did not change. The amended source-audit SHA-256 is
  `5a767ead523781dbd5dcec754515289cd65d1bdc12674e2bd82adf244815c580`.
- The publisher GigaDB 100241 file-inventory API links exactly
  `NFBS_Dataset.tar.gz` at the declared Wasabi HTTPS URL with byte length
  **1,751,464,473**. DataCite DOI `10.5524/100241` declares CC0-1.0. The
  publisher-linked mirror HEAD returned HTTP 200, the same Content-Length,
  opaque multipart ETag `"eda97fff56b5c9a8ec8f6ea3380e0b29-335"`, and
  named `x-amz-meta-md5chksum=BsRnrJq2y9uBheZkOHlyHQ==`; decoding gives
  expected whole-file MD5 `06c467ac9ab6cbdb8185e6643879721d`. The
  GigaDB file table's own MD5 cell is blank, so the publisher-linked mirror
  metadata is the checksum authority; no payload checksum has yet been
  observed. The multipart ETag is **not** treated as MD5. All six proposal-bound
  local source metadata files matched their SHA-256 values.
- Published **125 people** are prospectively one dependent TRAIN source
  family, with no internal SELECT/EVAL or external-test claim. The archive's
  person inventory is unopened and cross-source identity/pretrained exposure
  unknown. BEaST-derived label ancestry and exact seed membership remain
  unresolved; label QC and intended-use admission are deferred. The latest
  user steering permits synthetic and simulator-generated learning, so unknown
  ancestry is a provenance/dependence fact rather than a blanket generated-label
  ban. This source is not glioma, surgical action, brain deformation or force
  validation. A discovered alias with an
  existing protected person must retain that earlier role and be quarantined
  from TRAIN use.

## Transport review

- Runner binds the eventual committed cohort SHA, runner SHA, copied source
  files, one exact URL/size/checksum, one worker, and at most three durable
  attempt intents across restarts. It requires root release, frozen cohort
  status and role commit, the exact preceding ReMIND terminal receipt, and an
  exclusive ReMIND queue lock before starting. A disk check reserves 100 GiB
  after the remaining archive bytes.
- The reused transport stages a resumable partial. Its wrapper creates a
  certificate-validating TLS context, rejects redirects before a follow-up
  request, sends `If-Match` with the pinned opaque ETag, and requires the same
  effective URL, exact ETag, named MD5 metadata and remaining Content-Length.
  Resumed requests require HTTP 206 and exact `Content-Range`; the transport
  rejects compression and oversize bodies. A complete partial is checked
  locally without a new GET. Source length and full-file MD5 are checked
  **before** atomic publication; local SHA-256 and immutable per-attempt HTTP,
  intent, result and completion receipts follow. A prior integrity/scope
  refusal blocks further network retries, while transient transport failures
  retain the partial and count against the three-attempt bound.
- The runner never extracts the archive, decodes arrays, opens labels or
  updates an optimizer. Completion remains unreviewed opaque bytes. A hard
  kill may prevent a final completion receipt, but prior intents/partials
  remain for a bounded restart; this is not a scientific acceptance signal.

Before any GET, root must commit the frozen cohort role, rebind the declaration
to that exact hash and source snapshot, set a reviewed execution release, and
ensure the ReMIND terminal/lock gate is satisfied. A successful transfer then
requires observed full-file MD5 and size, plus saved local SHA-256; only a
later separate review may consider anatomy or intended-use admission.
