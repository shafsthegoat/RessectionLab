# Proposed V2: one quarantined image, no scientific release

Fix **RHUH-0001, preoperative visit 0, T1** before pixel inspection:
`/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz`.
The received index publishes `a5c579950d0431d04928e5b59e91ce9d`; its own SHA256 is
`7c6ba3fa767bf169679db30caa854e671796416ca365fac6e57662796177c130`.

This is a prospective exception for a separate acquisition experiment. V1 stays
unchanged and its prior-known-length/algorithm gates remain unmet. Request only
this source, once, through the already verified public FASP route. Retain at most
**64 MiB** in one compressed regular file, with a **60-second whole-action budget**
and 59-second hard watchdog. Record actual length afterward; never describe it
as matching a previously known length. No implementation or execution is released.

The inspected official client/schema establishes rate controls and size
precalculation, **not a native hard byte-cap option**. Proposed enforcement uses
an OS file-size limit plus one exact destination, isolated quarantine and final
file-count checks. This caps accepted image payload; it does not bound network
retransmissions or prove a quota for every transient side file. A strict transient
aggregate-storage requirement would need a separately reviewed OS quota.

Predeclare **MD5 of the complete original compressed bytes** as the sole candidate.
A matching published token establishes compatibility with that format/byte domain;
the publisher's algorithm remains unconfirmed. Nonmatch, timeout, extra files,
leftover writers or cap/client failure ends the experiment without retry or trying
alternative hashes. Record SHA256 and preserve partial/failing bytes as unaccepted.

Authenticated transport, compatible digest and verified image identity are
separate findings. Even a match stays quarantined until independent source/version,
path/index and byte reconciliation. No anatomy, modality, injury or training claim
follows from checksum compatibility.

Future header decoding requires a separate release: first read at most 544
uncompressed header bytes; cap offset/extensions at 1 MiB, total decompression at
256 MiB, scalar 3D dimensions at 512 each and 64 million voxels, and calculated voxel
storage at 255 MiB before allocation. Reject extra gzip members, truncation or
inconsistent bounds. Scientific use still requires subsequent image/grid review.
