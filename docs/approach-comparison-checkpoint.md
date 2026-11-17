# Native approach comparison and procedure experiment

## Resume

Work in the primary checkout on `shaf/next-changes-20261117`. The starting revision is
`aa41441f`. The verified external mount is `/Volumes/shafrir` with volume UUID
`9CC3623C-E38B-3EEE-B6BB-B9A06C193700` and data-root marker ID
`756c4d3b-aabf-4837-a354-52aa500af3f9`. Evidence for this experiment belongs in
ignored `testcases/approach-comparison-goal-v1/` or an indexed external location.

Check code and data identities before continuing the first unfinished milestone.

```sh
git status --short --branch
git rev-parse HEAD
export RESSECTIONLAB_DATA_ROOT='/Volumes/shafrir/old coding projects/RessectionLabData'
.venv/bin/python -c 'from resectionlab.data_root import checked_root; print(checked_root())'
```

## Milestones

| Milestone | State | Evidence |
| --- | --- | --- |
| Native comparison | Prepared and profiled, desktop verification in progress | `eden-source-verification.json` binds all six masks. CTRL_0933 was the first sorted remaining DEVELOPMENT host after CTRL_1277 and CTRL_0866. The first profile measured 129 approaches and 34 scenarios per finalist in 23.48 s with 1.29 GiB peak RSS. |
| Generated procedure | Cold engine replay verified, desktop verification in progress | `procedure-attempt-02/report.json` records a native protection refusal for inferior/posterior ingress. The superior approach treated 54 of 108 mm³, retained 54 mm³ and exited with checked STOP and zero source neural or bone removal. |
| Matched baseline | Frozen and running | `baseline-profile.log` records 181.87 s and 132 MiB peak RSS. `baseline-v1/protocol.json` freezes five conditions from four generated groups, four methods and two matched budgets. `baseline-v1/progress.json` tracks committed rows and budget. |
| Gated learning | Pending gate decision | Zero optimizer updates. The decision requires useful baseline progress, a measurable advantage to test and agreement between endpoint objectives and reward. |

The next baseline invocation reuses saved rows and charges new work.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u scripts/run_approach_procedure_baselines.py --profile testcases/approach-comparison-goal-v1/baseline-profile.log --output testcases/approach-comparison-goal-v1/baseline-v1
```

Acquired-scene entry, instrument work and exit under `surgicalProcedurePolicyV1`
remain pending. EDEN lacks reviewed complete negative coverage of critical
structures, independent working entry and access evidence, and tissue/effect
evidence across the whole two-hand action domain.

The source is acquired healthy anatomy with separately generated geometry. The
Mac is an M2 Pro with 16 GiB physical memory and 10 CPU cores. The installed
numerical backend selected C++ with one native thread. Its identity is
`304dea4eae193865a3fcb7e87a288ba58f4812174bfa70dcd7c6e9367e719e78`.
The procedure-aware resident executor remains ineligible, so the Python
engine owns procedure decisions.
