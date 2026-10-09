# Generated Conv3d memory optimization

An inference-only operator computes output-depth tiles using the complete input receptive field, preserving full outputs for later normalization. Six generated boundary/stride/dilation/grouped cases matched native Conv3d exactly. Independent inspection verified the receptive-field formula and compared the full saved outputs from a separate 16→16-channel, 32×48×48 FP32 experiment; they are byte-identical. No checkpoint or trained network was loaded.

In separate supervised single calls, native peak sampled RSS was **351,387,648 B** and tiled **241,319,936 B**, about **31% lower**. Both stayed within the fixed 1 GiB / 30 s limits with normal kernel pressure. Single forward times were 22.38 ms versus 10.11 ms; these exploratory times do not establish a repeatable speedup. The totals include the Python/Torch baseline.

This is generated operator evidence, not a full-network result, patient prediction or RL improvement. A separate full-network comparison is being prepared with unchanged weights, normalization, FP32 and declared output criteria. The failed v3 CPU attempt and unresolved original-size/full-128³ gates are preserved. An initial isolated-mode import-path failure is retained in the execution report.

Compact receipts and exact source snapshots are included; arrays stay outside Git. See `INDEPENDENT_REVIEW.md` for independently checked output and source hashes.
