# Eight controls after accepted runtime build

This is a **handoff only**. No active execution declaration, runtime identity,
model release or solver invocation is created here. Root's actual accepted
runtime identity is still required. The patch-identity SHA is not the runtime
identity SHA; the latter must bind the installed executable and libraries.

The generic runner passed 31 controls and remains unchanged. The final backend
source passed its separate repin review. Do not repeat 31 tests merely to replace
these pins; use the following narrow source/deck checks.

## After root sends the accepted actual runtime identity

1. Use the exact committed generation source recorded in `handoff.json` (or an
   explicitly reviewed replacement with identical bound source/deck bytes).
   The read-only **27-file** generation archive named in `handoff.json` is
   already prepared under `build/validation/`; its exact file hashes and tar SHA
   were checked without creating a declaration or model. Recheck those hashes
   before reuse. A replacement must use `generation_closure` from the same exact
   commit. This contains every final runner dependency except the not-yet-created
   declaration. Do not import mutable working-tree sources during generation.
2. From that archive, call its existing runner's `prepare` command with a fresh
   staging output directory. This writes eight deck copies and a declaration,
   and invokes no solver. Compare all eight staging decks byte-for-byte with the
   already committed solver-only copies. Require unchanged case order/caps,
   exact original-to-adapted solver subtree replacement, and declaration `sources`
   matching all eight archived Python source hashes. Check the final backend
   source hash and its combined patch/source constants against `handoff.json`.
3. Preserve the generation receipt: accepted runtime identity path/SHA, explicit
   parent acceptance, generation commit/archive path, generation argv, source
   hashes, eight equality checks and declaration SHA. Add **only the verified
   declaration** to `artifacts/mechanics-accelerate-controls-v1/declaration.json`.
   Never overwrite the existing decks, generic manifest or historical draft.
   Root commits this declaration/receipt before execution.
4. Create a separate fresh **28-file** Git archive from that new exact commit,
   under the same repository. Select `execution_closure`; verify every file
   against its exact Git object. This final immutable archive includes the new
   declaration. It is distinct from generation staging and actual output.
5. Write one new root-authorized release using the existing template: source
   commit and final archive, repository, one fresh attempt directory inside the
   repository, actual runtime-identity `{path, sha256}`, and root's explicit
   eight-control authorization. Do not reuse an old attempt/release. Check the
   declaration and runtime bytes without launching a worker if a final read-only
   preflight is desired; those methods perform no model solves.
6. Only after actual release, root invokes the final archive's existing command:

   `.venv/bin/python <archive>/scripts/mechanics_accelerate_controls.py run --release <release.json> --output <fresh-attempt>`

The fixed allowance is one 60-second supervised worker sequence, 3 GiB sampled
process-group RSS, one numerical thread, eight cases in order, four steps per
case, and zero retries. Stop at the first failure. Original physics, primitive
checkers and tolerances stay fixed; no automatic regeneration, larger cap,
alternate backend or retry follows a failure.

## Saved-output acceptance

Require the parent execution record to be completed, all eight actual backend
selection messages to select Accelerate, each executed XML to match the exact
solver-only transform, all primitive checks and stiffness scaling to pass, and
source/runtime bytes to remain unchanged. Both group summaries must bind the
same actual runtime identity and shared parent/results/baseline records. The
backend profile verifier rechecks bound raw evidence and original numerical
checkers; it does not accept summary booleans alone. A successful build or zero
worker exit is insufficient. Independent saved-output review follows the single
actual attempt; no solver rerun is implied by that review.
