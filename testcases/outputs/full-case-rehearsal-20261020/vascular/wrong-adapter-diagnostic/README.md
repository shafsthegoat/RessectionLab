# Excluded initial adapter diagnostic

The first local generator used `from_case(load_case(source))` directly. This skipped the source workspace's saved standardized environment mapping/profile and its strict protection contract. Its scene identity was `sha256:db234ba016fa8ab3d3111fa0fdd2290a4c0d01dd5a8df6d8949017f4535fa12c`, and its standalone rescue state identity was `sha256:78173fa51b3ef2374535f29df9bcf27647dd22e41da6c71bbbafbe7591af9fe1`.

Those initial generator filenames were overwritten by the corrected runs before this diagnostic directory was created. No original complete native/snapshot receipts were recoverable. This note reconstructs the adapter error from tool output; it is not a preserved or replay-verified original receipt. No result from that initial run counts toward the final exact-scene proof.

All current receipts in the parent directory were rerun using `load_environment`, the exact saved standardized scene `sha256:7a5d368382fe2a66192bdef88ff6af3a55cd533c9ba3787c68a1d1e3cb48f856`, and the matching strict protection contract. Future distinct versions must be kept separately rather than replaced.
