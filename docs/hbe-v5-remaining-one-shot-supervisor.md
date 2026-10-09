# HBE v5 remaining-row one-shot supervisor

This is a source-only preparation for the eleven frozen rows after the actual
`compression:N8:S60:reference` call. The tracked
`manifests/experiments/hbe-v5-remaining-one-shot-preparation-v1.json` has
`release: null` and all phase gates closed. It does not authorize a native
run, fitting, a held-out torque read, or physical validation. Each row needs
its own separately reviewed one-call release, at the current committed source
revision, in the original v5 order. A failed or interrupted attempt consumes
that row's sole output path; there is no retry or cap escalation.

The shared `scripts/mechanics_hbe_v5_remaining_one_shot.py` verifies the
complete executing source closure against one Git commit and exact SHA-256
file bindings. It checks the frozen v5 declaration, source-deck map, complete
native mesh, topology and boundary conditions, full adapted displacement
schedule, one repaired FEBio runtime/backend profile, one thread, and a fresh
row output directory before reservation. It rehashes the complete predecessor
chain and its closed output directories. The actual N8 receipt is mandatory
row zero, with its original release, deck, source commit and 61-frame
numerical decision. Every later predecessor also requires its preserved
one-row release bytes, runtime/profile binding, source identities, adapted
deck, and saved readout work-order token hash. The N8 receipt recorded
15,488,457 active bytes before the
final readout/receipt persisted; an independent inventory of the closed six
files is 16,075,117 bytes. The latter is charged to the aggregate output
budget. The historical receipt is preserved. N8 native time was
4.570366250118241 s; its separate readout and preparation times were not
recorded and are not imputed.

For each later row, the native call and complete-stream readout are separate
bounded process families. Native and readout each have a 3 GiB sampled group
RSS cap. Native wall caps by mesh are 90 s (N8/N12), 420 s (N16), 600 s
(N24), 1,800 s (N32), and 2,400 s (N36); S120 retains its N cap. The readout
cap is 600 s and the preparation/hash/preflight cap is 150 s per row.
Per-row output caps are in the preparation manifest, 64 MiB through 2 GiB.
These are prospective engineering resource ceilings, not scientific
tolerances or estimates of expected results. Existing completed,
exit-zero old-endpoint Accelerate controls informed the caps. Their observed
native seconds / saved-log bytes by frozen row 1–11 were:
`7.818/26,510,876`, `88.586/113,662,156`, `130.325/196,122,723`,
`639.057/457,432,281`, `873.335/648,565,508`,
`1,766.548/1,294,597,809`, `1.925/8,470,464`,
`7.240/26,162,239`, `20.725/59,428,081`,
`126.861/193,461,365`, and no qualifying analogue for row 11.
These old-endpoint observations do not predict new-endpoint behavior.
The all-12 native ceiling is 9,600 s
including the prior N8 reservation; eleven readouts have 6,600 s and eleven
preparation stages have 1,800 s. The eleven 150 s row ceilings sum to
1,650 s, leaving 150 s aggregate headroom that no row may spend to exceed
its own cap. The known-stage ceiling is 18,000 s. The
aggregate closed-output cap is 6 GiB; each preflight requires remaining cap
plus 2 GiB free storage. The per-row output reservations total 5,952 MiB;
with N8's 64 MiB reservation the sum is 6,016 MiB, leaving 128 MiB under
the aggregate cap. RSS and output size are sampled, so brief between-sample
peaks may be missed.

The supervisor requires exactly one Accelerate selection in both the native
console and solver log, no fallback text, exact executed adapted-deck bytes,
and complete native file inventory. It then launches the separately supervised
Python readout worker to replay *all* 61 or 121 saved frames and frozen v5
solver, force, moment, reaction, motion, energy/work and residual gates. The
readout's source/mesh/deck/primitive bindings and work order are tied back to
that one native attempt. A passing receipt binds all native and readout files,
and a quiet, zero-byte `readout-console.txt` is permitted only when its worker
exits successfully and the nonempty saved readout passes all checks. The deck,
native logs, work order and readout JSON remain mandatory and nonempty.
The receipt binds
the source before and after, release, runtime, prior receipts, actual stage
timings and numerical decision. The release commit must be HEAD before
launch. Later unrelated commits may advance HEAD while the row runs only if
all executing source files still match their released blobs at the immutable
commit. The final mutable receipt deliberately does
not assert its own exact final size or hash; the next release and independent
audit hash the complete closed directory, including the receipt, and charge
that measured size. No numerical result is a measured mechanical response,
patient validation, clinical probability, or permission to fit.

`tests/test_mechanics_hbe_v5_remaining_one_shot.py` exercises the frozen
order and caps, all eleven real source/mesh/deck adaptations without FEBio,
replay of the actual saved N8 grammar, missing release before reservation,
backend refusal, malformed readout identity/frames/solver, work-order
substitution, process-family wall/RSS/output kills, source closure, and
aggregate accounting. Existing `tests/test_mechanics_hbe_v5_stream.py` and
`tests/test_mechanics_hbe_v5_n8_one_shot.py` additionally mutate missing
frames, malformed frame records and residual evidence against the frozen
reader. Independent review of the source, first row release and output is
required before each next native call. The existing generated comparator
remains separate and must not silently promote these runs to fit or physical
evidence.
