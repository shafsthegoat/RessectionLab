Monitor-only manual retry candidate; original execution remains immutable

Attempt01 was interrupted by the parent allocation sampler after os.walk listed
report.json.tmp and the worker atomically replaced it before lstat. The native
model did not fail. Its progress receipt records11 calls started and11 returned,
0.6793625834397972 native seconds;1,416,400 leaf tests;23,224 attempted new leaves;
1,782 witness rows;38 peak tracked parents;zero preview budget refusals. The
parent sent owned-group SIGTERM; child exited-15 and was reaped, with no cleanup
errors or survivors. Parent elapsed1.5872435001656413s, peak84,885,504B. Original
partial report/geometry/progress/log/source index all remain in comparison-run-01.

Original progress SHA256:
871b2255176684fca86997e15d9f2a226ccc210e5ed7236ff971fde39ebc89b3
Original launcher SHA256:
1c6eb8c57543d39e8c9c533e7185bee8f6941c53377d677d4929e508d20b3ebe

The only runtime repair is in copied run_owned.py: a live output allocation
sample ignores FileNotFoundError only for the exact top-level atomic writer
temporaries report.json.tmp or progress.json.tmp. It does not ignore permission
errors, arbitrary vanished files/directories, or symlinks. Existing temporaries
count fully. Source scans and final scans after child reap remain strict, and
the full8MiB cap is unchanged. Partial report finalization now records a null
criterion and saved progress without KeyError; full acceptance still requires
a completed18-row report with a criterion dictionary and the original gates.
Every finalization exception explicitly resets acceptance to false, including
bad progress encountered after the main completed-report checks.

probe_exposure.py and test_comparison.py are byte-identical to attempt01.
config.json differs only in fresh output comparison-run-02. freeze_sources.py
additionally binds the two prior-failure receipts above. No model, criterion,
depth, cap, tool, action, reward, observation or target-array change. This is a
root-authorized manual retry for a monitor defect, never an automatic cap retry.
Any later total cost must add the prior11 completed calls to new attempt costs.

Twelve focused controls (no engine/process launch; root runs):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -I -B -m pytest -q -p no:cacheprovider build/native-cut-residual-diagnostic-v1/comparison-repair-v1/test_monitor.py
Expected below1s/128MiB. Controls cover two vanished owned temporaries, arbitrary
files, strict source/final scans, permissions, symlinks/counting and four fake
parent partial/missing-criterion/bad-progress/late-final-scan refusals. Successful
partial finalization preserves the saved11-call accounting without a criterion.

After root's passing controls and explicit release:
  .venv/bin/python -I -B build/native-cut-residual-diagnostic-v1/comparison-repair-v1/freeze_sources.py --release
Use printed exact SHA with a never-created parent cache prefix:
  .venv/bin/python -I -B -X pycache_prefix=build/native-cut-residual-diagnostic-v1/absent-comparison-parent-cache-02 build/native-cut-residual-diagnostic-v1/comparison-repair-v1/run_owned.py --source-index-sha256 PRINTED_SHA256 --execute
Limits remain240s worker/250s parent,10s cleanup,2GiB group RSS,one thread,
8MiB fresh output,180 native calls/18 audits,50GiB free disk. No execution by author.
