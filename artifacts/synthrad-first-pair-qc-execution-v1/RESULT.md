# SynthRAD Task1 first TRAIN-pair restricted QC

The one-shot, committed-source QC run opened only the frozen `1BA336` CT and
MRI archive members. Extraction, header, finite-scalar and source-coded grid
stages completed; the worker exited normally and the parent reaped it in
1.342 seconds. Peak worker RSS was 70,238,208 bytes, and all 20,705,884 new
output bytes stayed within the frozen limits. Both images have source-coded
shape 213×230×165, 1 mm spacing and matching coded affines. This does **not**
establish anatomical or scanner registration.

The terminal status is **`review_pending`**, not an intended-use pass. CT source
values span −1024 to 1929, but clinical HU scaling is unverified. The CT narrow
display window clips many source voxels; MR contains 55.0% exact zero
voxels. Restricted images still need privacy, coverage, calvarium and gross
CT/MR correspondence review. Overlap and source interpretation remain open.
No training, spatial-planning, scanner-frame or anatomical admission occurred;
no image preview was released from restricted local storage.

The [independent saved-output audit](independent-review.md) passed 87
receipt/metadata checks. It verified the committed source and declaration
chain, selected members, stage hashes, resource accounting, and the frozen
126-subject TRAIN denominator: one `review_pending`, 125 `not_attempted`.
It did not independently reopen image payloads or replay raw ZIP CRC and
content hashes. The [parent](parent-result.json) and [worker](worker-result.json)
receipts are exact copies of the local portable JSON; restricted images,
raw headers and staged gzip files remain outside Git.

This is one structurally processed, still-unreviewed patient-derived pair.
It provides no glioma anatomy, observed surgical action, vascular or functional
truth, validated tissue force, or patient-transfer result.
