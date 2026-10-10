# Portable fixed RL256 transfer candidate

This candidate connects the previously trained **aspiration-only** RL256 actor to the same generated 13×13×12, six-step native episode, action projection, restricted SEARCH, replay, workspace and post-seal vascular sidecar. The old checkpoint was trained on a different 9×9×7, two-step generated task. This is a distribution-shift software transfer, not mixed-mode learning, patient validation or a clinical comparison.

`promotion-files.json` maps 11 exact files (226,320 bytes) into `src/resectionlab` and `tests`. The existing tracked aspiration projection must retain SHA-256 `69d0003350f30fa03996a272137521ae19a0f511667872ffb45cc386d7b1132c`. The relocated `legacy_checkpoint_rollout_v3.py` is byte-identical to the committed prior source snapshot, SHA-256 `b48831841b27742d123d1338ef2ff05005f39f1d6b755617c2c7bc959ea749e4`. The pinned local checkpoint remains outside Git and is not in this package.

The supervisor and fixed loader now bind only sibling installed source files plus exact tracked core files. The worker also launches from the supervisor's sibling path. After the mapped bytes are copied unchanged, the actual execution path does not point into this ignored `build/` candidate. The loader compiles the exact checked prior source bytes, rebinds its checkout and model-asset roots, then supplies one exact-size, regular-file, SHA-checked byte payload to its unchanged scoped `weights_only=True` deserializer. No unrestricted pickle fallback exists.

Validation so far: seven no-checkpoint backend guards passed; one generated fake-identity actor/search/native replay passed; the pinned source closure passed; the preceding V2 staged development/vascular/workspace regression passed 42 tests. Independent V2 backend review passed 17 controls. The V3 portability delta awaits independent review. None of these checks loaded the real RL256 checkpoint or measured transfer performance. The fake-identity desktop fixtures are transport tests only.

After exact promotion and independent V3 review, root can execute one fresh attempt under the existing 20-second work, 25-second total operation/cleanup, sampled 1 GiB worker RSS, 2 MiB result and 1 MiB log limits. The controller has no automatic retry. A new result must be compared to restricted SEARCH on the same projected action inventory, and live authorship must become unverified when saved work is reopened. Previous V1 proposal defects and V2 review are preserved separately.

The root-owned one-shot command after promotion is:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I -B - <<'PY'
import hashlib
from pathlib import Path
from types import ModuleType
source = Path('src/resectionlab/legacy_transfer_supervisor.py')
if source.is_symlink() or not source.is_file() or source.stat().st_size > 128 * 1024:
    raise RuntimeError('Bound controller source is unavailable')
payload = source.read_bytes()
if hashlib.sha256(payload).hexdigest() != '60b25c4676954902e81ee32a1232f9e480573beb6e7a7dd8db337e890bca8b24':
    raise RuntimeError('Bound controller source changed')
module = ModuleType('fixed_reviewed_transfer_supervisor')
module.__file__ = str(source.resolve())
exec(compile(payload, str(source), 'exec'), module.__dict__)
module.run_attempt(Path('build/legacy-transfer-canonical-run-v1/attempt-1'))
PY
```

The attempt path must be absent; the supervisor reserves it before model work. Use a new declared path only for a separately reviewed successor, never an automatic retry.

This remains a local checkout release: the fixed checkpoint and archived training evidence must be present at the pinned relative paths. Ordinary packaged-app model asset distribution is a separate dependency.
