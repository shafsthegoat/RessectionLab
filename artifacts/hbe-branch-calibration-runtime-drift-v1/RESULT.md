# Interpreter drift before measured calibration

The v1 branch calibration passed its committed-source preflight, then its pinned Python executable was replaced locally at 2026-10-09 01:00:26 UTC. The supervised child refused the new SHA-256 before reading calibration or held-out curves, fitting a modulus or calling FEBio. The first result remains failed, with `physical_validation_pass` null and no retry.

The exact seven compact predecessor JSON receipts are separately tracked at their original paths (`build/` and `outputs/`), despite those directories normally being ignored. They total less than 15 KB and contain no patient payload, native solver logs, model weights or credentials. Their hashes are listed in `predecessor-bindings.json`; this allows the v2 source to verify the original path-and-content joins without rewriting prior receipts. The prior v1 release source tar and native experiment primitives remain local, with their separately recorded hashes.

The current FEBio backend profile and native runtime metadata still verify under the replacement Python interpreter without a solver call. This check does not establish cross-interpreter numerical equality; the bounded v2 experiment and independent physical comparison remain required. No measured curves were opened for this diagnosis.
