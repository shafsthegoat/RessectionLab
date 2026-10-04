# Existing FEBio sparse-backend audit

**The pinned runtime contains Apple's sparse backend, but its adapter has a blocking source defect. Keep the current Skyline memory caps.** No solver, installation, build, patient read, or runtime mutation occurred in this audit.

FEBio commit `32ae206ff4881dfb54f62296cd1558e58ed9fcc6` registers `accelerate`. The local build flags contain `HAS_ACCEL`; the link command includes the system Accelerate framework. Static Mach-O inspection finds the implemented adapter methods and an imported sparse solve symbol. The executable and `libnumcore`/`libfecore` hashes match the existing runtime identity. All 13 inspected source files match the original acquisition inventory. These establish compiled availability, not successful numerical validation.

The exact selector for a future symmetric direct candidate is:

```xml
<symmetric_stiffness>1</symmetric_stiffness>
<linear_solver type="accelerate">
  <iterative>0</iterative>
  <factorization>4</factorization>
  <order_method>0</order_method>
  <print_condition_number>0</print_condition_number>
</linear_solver>
```

The pinned enums map factorization 4 to LDLT with threshold partial pivoting and ordering 0 to AMD. The adapter accepts symmetric CSC input. Apple's [factorization documentation](https://developer.apple.com/documentation/accelerate/sparsefactorization_t) describes the symmetric family, and its [Sparse Solvers documentation](https://developer.apple.com/documentation/accelerate/sparse-solvers-library) describes the framework's sparse-system scope. Neither proves this particular FEBio adapter is correct.

## Blocking findings

`NumCore/AccelerateSparseSolver.cpp:202–203` allocates and copies `NonZeroes()` column-pointer entries. `FECore/CompactUnSymmMatrix.cpp:545` allocates that source array with only `Columns()+1` entries. A typical finite-element matrix has more nonzeros than columns plus one: the adapter reads past the source allocation. If the nonzero count is smaller, the destination cannot hold the complete CSC pointer array. This is a concrete static source finding, not a sanitizer reproduction; no attempt was made to execute it. A tiny successful solve would not clear undefined memory access.

The same adapter declares its iterative `rtol` and `atol` fields as integers but initializes them with fractional values. The existing build log contains truncation warnings. Direct mode does not use these fields; iterative mode should stay excluded. No source patch has been applied.

The registered MKL/Pardiso, Pardiso Project, Hypre and SuperLU paths are disabled or stubs under the recorded OFF build options. `cg` and `fgmres` require unavailable MKL solve paths. Built-in `bicgstab` is compiled, but its convergence/breakdown handling and nearly incompressible patient performance have not been validated here; it is not a cleared replacement. No paid solver or new physics engine is proposed.

## Smallest separate repair and validation scope

A separately authorized local patch can change the CSC offset copy/allocation to exactly `Columns()+1` entries, with a focused bounds regression covering sparse/dense profiles and the terminal offset. Zero-equation behavior needs explicit review. Preserve original source and the current runtime; bind the upstream commit plus patch in a new isolated source/build/install prefix. Reuse existing compiler, CMake and OpenMP. Prospective upper bounds are 120 seconds configure, 900 seconds build, two build jobs and 3 GiB process-family RSS; these are caps, not measured costs. No rebuild or installation was performed. Leave iterative mode disabled; correcting its types is a separate optional change if needed.

After the bounds fix is independently checked, reuse the existing `tet10_affine`, `mpc_translation`, and `mpc_nonrigid` control decks under a new declaration. Preserve mesh, material, loads, constraints, time grid and numerical tolerances; change only the linear-solver subtree. Retain the existing independent displacement/Jacobian, constraint, residual, energy and virtual-work checks. The historical launcher explicitly requires a Skyline selection message, so a new candidate launcher must bind and require `accelerate` rather than weakening the old guard. Freeze both manifests, deck hashes and new runtime libraries before release.

The existing controls use six tet10 elements and 27 nodes. A prospective candidate run can keep the existing limits: three cases, four increments plus rest, one numerical thread, 60 seconds aggregate, 3 GiB process-family RSS and no retry. Stop on the first failure. Passing these controls would establish only a small software check; a later separately frozen bounded mesh study is needed before any scalability claim. The current specimen Skyline protocol and all its outputs remain unchanged.

## License and evidence boundary

The pinned FEBio adapter and source license are MIT, including commercial-use permission with notice retention. Accelerate is a dynamically linked system framework under separate [Apple software terms](https://www.apple.com/legal/sla/docs/macOSTahoe.pdf); it is not part of FEBio's MIT grant. Use the system dependency, preserve existing OpenMP notices, and do not infer permission to redistribute Apple's framework. This is an inventory of the current local build, not app-distribution clearance.

`review.json` binds the exact source/runtime hashes, flags, link command, prospective caps and findings. `binary-inspection.json` records read-only `otool`/`nm` results. The original source archive/inventory supplies the inspected code; no duplicate source tree is added. Root and the patient-mesh owner received the blocker before any backend selection.
