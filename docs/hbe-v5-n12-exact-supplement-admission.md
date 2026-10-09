# Exact N12 saved-attempt numerical admission

The first N12 native solve and complete-stream readout exited successfully, but
the parent marked its receipt `failed_or_incomplete` because the readout worker's
console was correctly empty. That receipt and its original status remain
unchanged. The later one-shot supplement replayed the saved native output
without another FEBio call. An independent audit approved that replay as a
**numerical-only** supplement, with an explicit gap: the original parent did not
record its post-run source and runtime guards after the packaging failure.

`mechanics_hbe_v5_n12_admission.verify_exact()` authenticates the one original
receipt and one supplement by full SHA-256, their separate releases and immutable
source commits, the original frozen runtime, complete saved output inventories,
byte-identical replay JSON, and the independent review artifacts. Only this
reviewed pair can enter the shared HBE v5 predecessor chain. Other failed
receipts remain rejected. The N12 readout is an **eligible future native
comparator row input**; no twelve-row native comparison has been completed.
The existing generated-fixture comparator retains its generated-only contract.

The frozen twelve-row study ledger charges the original N12 native call once,
its original readout and preparation time, and its 50,501,397-byte closed
directory. A separate bounded ledger charges the supplement's one saved-output
replay, 2.8474308329168707 s replay time, 3.4362154591362923 s preparation,
and 557,392 bytes. The combined N8-plus-N12 total is 25.71928400127217 s
and 67,133,906 bytes, with exactly two native calls. The supplement is not
quietly excluded from combined totals; it also does not retroactively change
the prospective 18,000 s/6 GiB twelve-row study cap. These limits govern local
resources, not physical or clinical accuracy.

This exception does not open fitting, held-out torque responses, physical
validation, measured or patient data, or any patient-specific inference.
