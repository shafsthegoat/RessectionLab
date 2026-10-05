# RHUH checksum-only transfer preparation

The helper is prepared, **not released or executed**. Its default command checks
local dependencies only. After root review, `python3 scripts/acquire_rhuh_checksum.py
--execute` obtains a fresh public authorization and requests only
`/RHUH-GBM-nii-v1.sums` from package 684. It makes one transfer attempt into a fresh
ignored quarantine directory. It requests no patient images.

The provider resolves that package-relative request to an absolute server path.
The initial literal comparison correctly stopped before transfer. A metadata-only
follow-up established the stable mapping; the helper now pins its SHA256 and
checksum filename. A subsequent fresh metadata request passed every guard. Both
the negative and corrected receipts are retained. No source mapping or transfer
credentials are saved in plaintext receipts.

The native signed client, bundled runtime license, and IBM-published common RSA
bootstrap key are pinned by hash. The public key is the standard client component
used by IBM's token authentication flow; it grants no data access without the
provider-issued token. The key remains in ignored build storage and is not
committed. Token and cookie go only into the child environment. Raw errors and
client output are discarded; transfer logging is directed to discarded stderr.

The child has an OS-enforced **65,536-byte file ceiling** and no core dumps. A
supervisor covers local checks, metadata, client work, and receipt writing in
a 57-second worker allowance, then always kills/reaps its process group. A
59-second hard watchdog also covers final readback; timeout exits 124. One
direct client invocation is permitted, with no application retry; no claim is
made about protocol retransmission. Only one regular file of exactly
**61,787 bytes** can pass validation; all failures remain unverified in quarantine.
Success records a local SHA256, not an independent publisher checksum claim.
Any surviving descendant after worker exit rejects success. Otherwise the
supervisor reconciles bounded receipt/payload bytes after group cleanup. The
worker receipt alone never authorizes use: supervisor `accepted=true` is required.

Twenty-three offline guard tests passed, including an actual OS file-limit
failure, credential separation, altered sources/hosts/ports, extra files,
symlinks, byte mismatches, and error redaction. The local preparation check
passed. Independent review added 14 controls; all 37 passed in 0.15 seconds.
Three lifecycle failures were preserved before repair. Root release is still
required for checksum transfer; image acquisition remains disabled.
