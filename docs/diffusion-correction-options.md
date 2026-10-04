# Local diffusion correction: dependency and feasibility audit

Initial audit, October 4, 2026: inspected primary documentation, source, release metadata and a 62.9 kB Python wheel, without installing or executing it. A subsequent isolated optional-runtime experiment is now recorded in [pyhysco-phantom.md](pyhysco-phantom.md): the known-field case improved relative image RMSE from 9.6788% to 0.4032%, with 0.3948 mm field RMSE, 9.34 seconds wall time and 415.2 MiB sampled peak RSS. Its zero-field solver failure and unsupported rotated-PE negative control remain rejected. These are synthetic results only; no patient correction or acquisition-metadata change followed. See `btc-acquisition-physics.md` and `diffusion-pipeline.md` for the remaining physics and reconstruction gates.

**Recommendation:** retain the pending FSL decision for the established complete correction path. The first FSL-free PyHySCO phantom experiment is complete; its explicit geometry gate remains closed for BTC. Consider AFNI as a more mature susceptibility comparator. Neither is a verified replacement for the combined motion, eddy-current, susceptibility and outlier correction needed for BTC. Unmodified TORTOISE V4 does not remove the FSL dependency.

## Options

| Candidate | Verified scope and license | Mac/local feasibility | Decision |
|---|---|---|---|
| TORTOISE V4 / DR-BUDDI | Main source GPL-3.0; the CPU executables link bundled FSL BET. DIFFPREP handles motion/eddy; DR-BUDDI handles susceptibility. | Current instructions recommend Linux binaries or Linux/amd64 Docker for macOS; native Apple Silicon build unverified. Release archive is 744,715,009 bytes. | Does not bypass the FSL license question. |
| PyHySCO 0.0.4 | GPL-3.0; opposite-PE susceptibility model implemented in Python/PyTorch. No FSL dependency found in the wheel or its declared dependencies. | Universal wheel is 62,904 bytes; dependencies already available locally. Subsequent isolated CPU phantom run: 9.34 s, 415.2 MiB sampled RSS for known distortion. | Same-grid phantom adapter tested; zero-field solver failure and BTC's nonparallel PE geometry remain gates. |
| AFNI `3dQwarp -plusminus` | NIH public-domain work, MCW CC BY 4.0 portions, and enumerated third-party licenses. Susceptibility-oriented symmetric nonlinear registration. | Official macOS 12+ ARM instructions exist. ARM archive HEAD reports 864,018,288 bytes; standalone `3dQwarp` 6,838,868 bytes, with runtime libraries still to inspect. | More mature comparator, but broader install and still not full DWI correction. |
| Existing DIPY motion correction | BSD-licensed rigid/affine registration; no reverse-PE susceptibility solver or EDDY-equivalent model established here. | Already installed on this Mac. | Preserve separate motion-only status; never promote it to complete correction. |

GPL permits commercial use and private execution; distribution introduces source/license obligations. That differs from an academic/noncommercial restriction. A main repository license does not supersede bundled component terms. [PyHySCO license](https://github.com/EmoryMLIP/PyHySCO/blob/bff665515231bfbd53165e8e18cfdc9c7fe38cf8/LICENSE), [AFNI license inventory](https://github.com/afni/afni/blob/cadf2054b5f12de7e451c2dd35c69c9ac08624b3/LICENSE.txt), [DIPY license](https://github.com/dipy/dipy/blob/master/LICENSE).

## Why TORTOISE does not resolve the dependency

Pinned source: `QMICodeBase/TORTOISEV4@1ffb8f362e8e8487d7a82cb8310d2532a9da5253`. Its CPU CMake branch links `external_libraries/bet/Linux/libbetokan.a` into **both** `TORTOISEProcess` and standalone `DRBUDDI` (lines 208 and 211). Supplying our own mask would not remove that static linkage. A custom BET-free fork would require new build, dependency and numerical validation; it is not an already verified alternative. [Pinned build source](https://github.com/QMICodeBase/TORTOISEV4/blob/1ffb8f362e8e8487d7a82cb8310d2532a9da5253/TORTOISEV4/CMakeLists.txt#L203).

The documented CPU build uses ITK 6.0 beta, Boost 1.86, Eigen 3.3, FFTW3, zlib and CMake; VTK and CUDA can be disabled. Linux static/GCC flags and a Linux BET archive still occur in that branch. The current README's macOS path uses Linux/amd64 Docker, not a verified native ARM CPU binary. The published 4.1.0 archive is approximately 710 MiB compressed; unpacked size and peak RAM are unmeasured. [Build instructions](https://github.com/QMICodeBase/TORTOISEV4/blob/1ffb8f362e8e8487d7a82cb8310d2532a9da5253/README.md), [release](https://github.com/QMICodeBase/TORTOISEV4/releases/tag/4.1.0).

DR-BUDDI's structural-input help requires T2-like contrast and explicitly excludes T1-weighted images. BTC's verified T1 must not be passed as a substitute T2. Motion/eddy and susceptibility are distinct processing stages in the [TORTOISE V4 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12690295/); the [structural parser](https://github.com/QMICodeBase/TORTOISEV4/blob/1ffb8f362e8e8487d7a82cb8310d2532a9da5253/src/tools/DRBUDDI/DRBUDDI_parserBase.cxx) supplies the exact input restriction.

## PyHySCO: useful experiment, necessary adapter work

Pin the [0.0.4 wheel](https://pypi.org/project/PyHySCO/0.0.4/) with SHA-256 `589e6509602b98b5f275d6e772c8ead8c8641ed1f4f20e2e24efaefb022bd884`. Author repository HEAD inspected: `bff665515231bfbd53165e8e18cfdc9c7fe38cf8` (April 22, 2024). The [primary paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1406821/full) evaluates CPU and GPU implementations against simulated and healthy HCP data. It does not establish tumor-case correction accuracy or runtime on this Mac.

Source inspection identifies four concrete integration traps:

- The wheel CLI requires positional `file_1 file_2 ped`; the README's named options do not match that parser.
- Loading uses the first image's affine for spacing, without enforcing agreement with the second image's affine. The NIfTI loader accepts PE axes 1 and 2 despite the CLI advertising 3 as well.
- `save_data` writes an identity affine. Its outputs cannot be imported as patient-space evidence without a tested adapter that preserves the reference grid. The field uses an augmented PE grid, so copying the image affine onto that field without deriving its sampling origin is also unsafe.
- The model fits equal/opposite axis distortions. It does not accept separate general PE vectors, perform motion correction, rotate diffusion gradients, or model eddy-current/outlier effects. Two PA b0s do not provide reverse-PE diffusion-weighted counterparts for all 102 AP volumes.

These findings come from the inspected wheel's `scripts/pyhysco.py`, `EPI_MRI/utils.py`, and `optimization/EPIOptimize.py`; the [pinned source loader/writer](https://github.com/EmoryMLIP/PyHySCO/blob/bff665515231bfbd53165e8e18cfdc9c7fe38cf8/src/EPI_MRI/utils.py) retains the geometry limitations. Audit files reside in `/tmp/ressectionlab-correction-audit/PyHySCO-wheel`; they are inspection artifacts, not installed product dependencies.

The subsequent experiment supersedes the original audit's illustrative direct-CLI command. Use its isolated runtime and reproducible driver, which include source-integrity checks, a header adapter and a watchdog; keep the optional package outside the app environment. With the verified wheel available:

```sh
.venv/bin/python -m pip install --no-deps \
  --target data/optional-runtimes/pyhysco-0.0.4 /path/to/PyHySCO-0.0.4-py3-none-any.whl
.venv/bin/python scripts/experimental_pyhysco_phantom.py \
  --wheel /path/to/PyHySCO-0.0.4-py3-none-any.whl \
  --output outputs/pyhysco_phantom/new-run
```

The output directory must not already exist. The driver generates phantoms and accepts no patient-image input. See [pyhysco-phantom.md](pyhysco-phantom.md) for the completed run, strict geometry rejections and retained solver failure.

## AFNI: mature comparator with a larger dependency footprint

Pinned source: `afni/afni@cadf2054b5f12de7e451c2dd35c69c9ac08624b3`. Current licensing includes the May 2026 MCW CC BY 4.0 update; do not rely on older blanket descriptions. Individual bundled components still have their own notices, so product packaging requires an inventory of what is actually shipped. [Current license](https://github.com/afni/afni/blob/cadf2054b5f12de7e451c2dd35c69c9ac08624b3/LICENSE.txt).

`3dQwarp -plusminus` estimates symmetric warps for opposite-blip images. Inputs must share a 3D grid; prior rigid alignment is a separate operation. `-noXdis -noZdis` constrains movement to the dataset's second coordinate, not automatically to arbitrary physical AP. This is a registration model rather than a full DWI eddy/outlier pipeline. [Program help](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dQwarp.html).

Future same-grid phantom comparison after installation verification:

```sh
OMP_NUM_THREADS=2 /usr/bin/time -l 3dQwarp \
  -base phantom_ap_same_grid.nii.gz -source phantom_pa_same_grid.nii.gz \
  -plusminus -noXdis -noZdis -minpatch 9 \
  -prefix outputs/afni_phantom/run1
```

The official [ARM installation instructions](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/steps_macOS_12_Silicon.html) install build/runtime dependencies and include administrator steps. A small headless subset may be possible, but has not been dependency-audited or executed here. Source recipes reference Xcode, XQuartz, Homebrew libraries and OpenMP; no claim of a 6.8 MB self-contained app follows from the standalone executable size. Archive sizes above came from HTTP HEAD only; neither archive was downloaded.

## BTC physics and bounded validation

The source audit establishes a 0.8-degree AP/PA prescription difference and supports a separately derived effective readout of `0.0361003696677854` seconds. The originals retain `0.0266003`; no derived sidecar has yet been created. Regridding PA into native AP space leaves a PE vector approximately `[-0.01396218, 0.99990252, 0]`, which is not exactly opposite AP's `[0, -1, 0]`. PyHySCO and one-axis AFNI half-warps do not represent that pair exactly. Do not erase this difference or interpret header prescription as measured head motion.

The subsequent `96 × 96 × 60` phantom experiment checked zero distortion, a smooth invertible field, opposite sign and the measured 0.8-degree grid/PE difference, alongside coordinate, intensity and header contracts. Its negative control shows why improved image similarity cannot overrule unsupported geometry. New correction implementations still require known-field tests and independent validation before any patient adapter. A flexible registration improvement cannot certify lesion-region anatomy.

A float32 image at that grid is 2.11 MiB; all 102 AP volumes alone are about 215.2 MiB. These are array sizes, **not measured solver peak RAM**. The subsequent phantom driver used one pair, two CPU threads, float32, a 90-second timeout and an 8 GiB RSS watchdog. Its known-field sampled maximum was 415.2 MiB; sampling can miss a brief high-water peak. This measurement does not guarantee a patient run's resource use or scientific validity. macOS memory limits alone are not a reliable RSS guard.

Persist each correction component's status separately. A future accepted result needs source/derived metadata hashes, algorithm/version/parameters, complete image and PE transforms, gradient rotation provenance, residual and Jacobian summaries, independent visual review and the remaining motion/eddy/outlier limitations. Until those exist, BTC tract evidence remains unaccepted.

MRtrix `dwifslpreproc` is not a FSL-free alternative: its [official command documentation](https://userdocs.mrtrix.org/en/latest/dwi_preprocessing/dwifslpreproc.html) uses FSL correction tools. MATLAB HySCO/ACID adds a runtime requirement not established on this host. No paid runtime, container service, infrastructure or billing change is proposed here.
