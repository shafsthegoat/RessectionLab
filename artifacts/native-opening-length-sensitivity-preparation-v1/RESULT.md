# Length descriptor diagnostic: preparation

The frozen experiment changes only four non-STOP working-length descriptors on
one exact generated root. It permits two frozen forwards, one per checkpoint,
with saved baseline scores and no physical action certification. Limits are
20 seconds, 1 GiB sampled RSS and one CPU thread; zero actions or updates.

Source commit `dcf8946` follows 21 focused controls and eight independent controls.
Two independent failures exposed an active method left marked running after a
terminal exception. Root repaired both first/second-checkpoint cases; completed
earlier output remains intact. Original failing source, logs and receipts remain,
as does the owner's initial fixture failure. No experiment forward ran during
these controls. The later actual run is recorded separately.
