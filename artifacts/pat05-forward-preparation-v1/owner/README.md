# PAT05 frozen-forward preparation evidence

The owner ran the focused software tests and metadata-only preflight before the source freeze. Original output was captured only in the conversation tool transcript; no original pytest/JUnit logs were written. `verification.json` accurately transcribes the observed counts, elapsed times and test-fixture failure, and identifies that limitation.

The first test attempt had 14 failures, 4 passes and 1 skip because the tiny analytical fixture reversed two positional tool fields. After using named fields, 18 tests passed with the not-yet-written manifest test skipped. The final frozen implementation passed all 19 tests in 3.64 seconds, and metadata-only preflight passed.

This directory packages exact frozen source/manifest copies and hashes. Packaging ran no tests, patient decoding or model calls. The unit tests used small analytical inputs and deliberately substituted test-only checkpoint authority; the two actual PAT05 checkpoint forwards remain a separately released execution.

Only the four files listed in the verification receipt belong to this implementation. Existing numerical sources and default guards were unchanged. The PAT05 bundle and raw RL checkpoints are local, hash-bound runtime dependencies and should remain outside Git.
