# Architecture evidence — October 4, 2026

The revised master plan remains the specification. This log records measured
starting conditions and reversible implementation decisions.

## Starting evidence

- The repository initially contained only the three specification documents and
  two commits on `main`; no existing application or experiment was found.
- A similarly named Codex output directory contained empty `work` and `outputs`
  directories. A bounded search of likely project locations found no other
  RessectionLab checkout. This does not establish that no copy exists anywhere.
- The machine is an Apple M5 MacBook Pro with 10 CPU cores, 10 GPU cores, 16 GiB
  unified memory, macOS 26.6, and approximately 733 GiB available storage at the
  initial inspection. No CUDA assumption is justified. Metal support was reported
  by macOS; a PyTorch MPS workload has not yet been tested.
- System Python is 3.9.6. The available bundled Python is 3.12.14 and includes
  NumPy, pip, and the standard `venv` module. The first probe found no Qt, VTK,
  PyTorch, SciPy, NiBabel, SimpleITK, or DIPY in that runtime.
- Swift 6.3.3 and Command Line Tools are installed; a full Xcode installation is
  not selected. Slicer is not installed in `/Applications`. Bundled Node 24.19.0
  exists, but Node, `gh`, and `uv` are absent from the default shell path.
- `scripts/probe_hardware.py` reproduces a privacy-conscious runtime inventory.
  `docs/hardware.json` captures the bundled Python environment before dependency
  installation; package absence there does not describe a subsequently built venv.

## Initial desktop decision

Prototype a local Python core with PySide6 and VTK in an isolated Python 3.12
environment. This is an initial implementation choice, pending real-case results.
The available runtime supports an immediate prototype without a source build of
Slicer. Qt documents macOS application-bundle deployment through its
[Python deployment tools](https://doc.qt.io/qtforpython-6/deployment/index.html).

The competing Slicer approach remains viable and has established imaging tools.
Its official [macOS build instructions](https://slicer.readthedocs.io/en/v5.12.0/developer_guide/build_instructions/macos.html)
describe a separate build toolchain and an experimental Apple Silicon recipe.
The absence of Slicer here is a setup observation, not a performance result.
No head-to-head Slicer/PySide comparison has been run. Downloading a prebuilt
Slicer application may avoid a source build and is the appropriate first fallback.

Keep imaging, geometry, search, simulation, and evaluation independent of the
desktop shell. Start with CPU algorithms and small policies; measure wall time
and memory before enabling parallel training on a 16 GiB shared-memory machine.
Prefer shared static arrays and bounded worker counts over one volume copy per
rollout. Population pretraining remains later work.

Revisit this choice if the prototype cannot reliably provide aligned MPR/3-D,
responsive cancellation, cavity replay, save/reopen, and a runnable Mac bundle.
Record actual timings and failures when those tests exist. Neither installation
success nor a synthetic screenshot completes the required real-case comparison.

## Reproduction entry point

Use a Python 3.12 interpreter to create the project `.venv`, install the project's
declared dependencies there, then run:

```sh
.venv/bin/python scripts/probe_hardware.py --output docs/hardware-current.json
```

The initial audit used the bundled Python reported by the desktop dependency
tool. No cloud resource was created and no paid service was used.
