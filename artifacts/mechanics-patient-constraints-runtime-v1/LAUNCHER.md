# One-shot launcher

The callable launcher is `scripts/mechanics_patient_constraints_run.py`. Root fills [release-template.json](release-template.json) only after committing and archiving the reviewed source. The release binds one absolute `attempt_directory`; attempting another output path is rejected, and an existing attempt directory cannot be reused. The archived launcher compares its closed nine-file numerical/runtime closure against the exact Git commit, checks the fixed manifest and accepted runtime identity, binds every listed executable/library/OpenMP/acceptance receipt, then delegates one worker to the unchanged process supervisor. The worker revalidates the same release and original inputs before invoking any case; both worker and parent record final input checks, including missing-file failures. Setup provenance checks precede the supervised60s sequence; the worker's repeated validation, all three solver calls and all output checking fall inside that allowance.

Future command, only after root's separate release:

```text
<python> <immutable-archive>/scripts/mechanics_patient_constraints_run.py run --release <released-record.json> --output <the-single-released-attempt-directory>
```

`worker` is an internal supervised entry point; it requires an untouched initialized result, the exact revalidated baseline/release and matching attempt directory. No runtime acquisition, build, mesh or arbitrary case is exposed by this launcher. Mocked tests execute no subprocess and retain the original test logs; runtime success still requires the subsequent actual attempt.
