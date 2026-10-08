# Exact-file RESECT TRAIN acquisition preparation

The separate importer covers the frozen 25 original-ultrasound/cavity-mask pairs
from 13 labeled TRAIN people. Case11 and the absent Case15 during label stay
missing within the 14-person TRAIN cohort. Protected Case4 and all SELECT and
evaluation cases remain outside its whitelist. This preparation does not change
the existing Case3 pilot or shared transfer helpers.

Each explicit invocation attempts one declared pair within 300 seconds, with a
295-second worker ceiling and a maximum pair size of 40,350,122 bytes. Sources
are bound to exact original URLs or individual OSF revision-2 routes, sizes and
published hashes. The ten short OSF aliases are specific to their declared file;
they do not authorize arbitrary same-host downloads. Existing macOS TLS trust
is retained. There are no automatic retries. Per-file receipts preserve a
successful first file when the other fails; continuation needs a new run ID and
an explicit link to the closed failed attempt. Acquired bytes have no QC,
anatomical, planning, training or RL admission.

Independent review initially reproduced four counterexamples: completion with
an empty pair, duplicate file IDs, a failed file, or final parent accounting over
the time limit. Their source, tests and logs remain unchanged. Completion now
requires the exact ordered pair with verified per-file immutable receipts and
source/claim bindings. Final accounting over the declared limit records timeout,
even if the worker completed. Actual isolated Case3-cache runs verified this
deadline behavior and subsequent explicit continuation.

The repaired implementation passes **69 owner and six independent controls**
(75 total, no skips), including a fresh copied-checkout metadata preflight.
The original Case3 compressed files remain unchanged; tests performed no
network access, new acquisition or decompression. Final receipt publication is
after the acceptance clock reading, so arbitrary storage stalls are outside
that timing guarantee. Exact rights bytes are still required in the local cache.

The next released operation is the first new TRAIN pair, followed by independent
source verification and separately bounded image/mask QC. A successful local
test does not establish current provider availability or anatomical validity.
The noncommercial annotation restriction remains attached to each source.

```sh
.venv/bin/python scripts/resect_train_intake.py preflight
.venv/bin/python scripts/resect_train_intake.py acquire --execute \
  --manifest-sha 09a15ffe52af966c27d53fba4924a7c67fba5590b12e53cb48aa022ef60f04f1 \
  --pair Case2-during --run-id train-Case2-during-01
```

Historical review scripts retain their original local paths. Their saved
negative results verify software boundaries; they are not fabricated patient
records or scientific datasets. Patient payloads stay outside Git.
