# Isolated outcome-baseline runtime

These four optional binary distributions were installed only in
`build/rhuh-baseline-runtime-v1/site-packages`; the shared project environment
was not modified. The existing Python 3.12.14, NumPy 2.5.3 and SciPy 1.18.1 are
reused. `receipt.json` records every installed non-cache file and the public PyPI
wheel URLs, sizes and verified SHA-256 values. `requirements.txt` pins the exact
macOS arm64 wheels used here; these are research runtime additions, not app dependencies.

Reconstruction from the repository root, using a fresh target directory:

```sh
.venv/bin/python -m pip install --no-deps --no-compile --require-hashes \
  --target build/rhuh-baseline-runtime-v1/site-packages \
  -r artifacts/rhuh-baseline-runtime-v1/requirements.txt
```

Run the declared research worker with that directory and `src` in `PYTHONPATH`.
Regenerate and verify a local runtime receipt before a new execution; installation
metadata may differ across machines. The study release separately binds the exact
interpreter, source and runtime receipt rather than assuming this path is sufficient.

The initial import failed because the new scikit-learn release also requires
Narwhals. That failure is retained; adding its pinned wheel resolved the import.
One four-row analytical fixture then passed monotonicity, symmetry and finite-fit
checks, with one OpenMP thread. It read no patient records. The known L2-parameter
deprecation warning is recorded in `verification.json`; no optimizer settings were
silently replaced. No RHUH model has been fitted by this runtime check.
