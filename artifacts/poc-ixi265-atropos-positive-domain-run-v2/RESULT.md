# Positive-domain fitting failed at the native N4 boundary

One separately released v2 attempt passes the unchanged source/frame checks and
the exact eight-cell missing-intensity partition, then terminates with SIGSEGV
after entering native N4. No N4 completion, Atropos entry, class maps or candidate
is recorded. No retry, tuning, anatomical acceptance or growth run follows.

The [sealed outcome](outcome.json), SHA256
`dce67f0acde21aa11fa99bdaa270108c8b53b5a091514182870e69e01fa78e21`,
retains7.68s and493,420,544 bytes sampled supervisor/worker-family RSS. Minimum
available memory is7,271,006,208 bytes; no resource limit fired. The source T1,
original brain mask, host bundle and all v1 failure evidence remain unchanged.
The full local OS report is excluded from Git; its hash and task-specific native
trace are retained in [the crash summary](native-crash-summary.json).

The [independent failure audit](../ixi265-positive-domain-independent-v2/result-qa.json),
SHA256`9c90b165a5da5657394ee1d26f0aa35193bc072e627b3b198d5fea1018beae72`,
verifies source-byte continuity, declarations, both protocol denominators and
the matching worker/parent IDs and ANTs native-library UUID. It records N4 entry
from the log and infers preceding guard passage from pinned execution order.
Its37,961,728-byte sampled peak involves metadata/hash checks only, with no
patient decode, ANTs import or solver execution.

The two declared N4/Atropos protocols therefore have two actual attempts and
zero successful or accepted results. The three older SynthSeg memory failures
remain separate and retained. The45 prospective software checks remain valid
within their scope; they did not establish native-library compatibility.

Source inspection suggests a mismatch between our unsigned-byte in-memory mask
and N4's float mask pointer contract. That is a hypothesis for a separately
declared generated-image diagnostic, not a cause proven by this unsymbolicated
crash report. Neither source intensities nor the anatomical mask is changed to
work around this failure.
