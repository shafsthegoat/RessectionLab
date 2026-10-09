# Prospective identity-bound model-monitor protocol

The original negative tile1 attempt is preserved separately in
`artifacts/tile1-cleanup-negative-v1/`. Its worker's logits matched the
saved tile4 array byte-for-byte, but its supervisor failed in cleanup before
writing a resource receipt. The original attempt remains unaccepted and is
not reconstructed here.

The successor keeps the same generated input, checkpoint, worker, output tile,
and 3 GiB sampled RSS / 120 second / macOS pressure guards. It changes only
how a child is released and finalized: a tiny launch gate waits for a byte
while the supervisor binds the kernel PID/start identity and process group,
waits for a clean child-excluded inventory, rechecks host pressure, and writes
the launch receipt. Closing the gate without the byte exits before the model
worker. After release, the existing monitor policy feeds the independently
reviewed fail-closed finalizer. No bare `killpg` call remains. A PID identity
check immediately before a signal narrows but cannot eliminate the Darwin
check-to-signal race.

Generated-only tests: `python3 -m unittest -v test_launch_control.py` passed
8/8. The copied finalizer helper passed 16/16 generated adversarial tests
before release. All tests use small generated processes and
never load weights, patient scans, or a model.

The parent waited for independent exact-source approval before the single
model call recorded in `RESULT.md`. This protocol applies to generated
software/resource evidence only, with no patient or clinical claim.
