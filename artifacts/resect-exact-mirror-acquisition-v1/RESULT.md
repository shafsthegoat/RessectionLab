# Exact RESECT mask mirror acquisition and offline import preparation

All 24 missing TRAIN masks were downloaded continuously into isolated staging in 9.1224 seconds: 1,561,749 bytes across 48 verified-TLS HTTPS requests. Every body matches the frozen original OSF revision-2 SHA256, MD5 and byte count. The original Case3 mask was already acquired and is excluded from this transfer.

The transport is the third-party MedOtter/RESECT-SEG repository at pinned commit e86fb37dd93f7a9c64e48952f71410af59b04b9b. Author-operated status is unconfirmed. Original OSF identifiers, rights, versions and checksums remain the source authority. The CC-BY-NC-SA-4.0 restrictions are preserved. No new agreement, OSF request, scientific decode or admission occurred. Signed redirect query values are not retained.

The original provider returned three 429 responses for Case2; those immutable failures remain. Root independently rehashed all 24 staged bodies, confirmed the original queue had no child transfer, and gracefully stopped its now-unneeded cooldown wait for a locked offline cache handoff. No transfer was interrupted.

The importer verifies all metadata, roles, staged bodies and existing destinations before publication. It holds the canonical cache lock, installs verified private copies without overwrite and retains per-file provenance and interruption receipts. Twenty-four focused controls pass, plus the actual read-only staged-body check. This milestone prepares import; the actual installation receipt will be recorded separately. The old queue's exhausted Case2 outcome is not rewritten into an OSF success.

Raw NIfTI bodies remain local and ignored. The archived acquisition script and source snapshot preserve the executed code exactly. To reproduce its original layout, restore the archived metadata to the corresponding build paths in declaration.json before running download.py there. The offline importer supports the portable metadata archive directly via its default evidence root.
