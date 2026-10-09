# One supervised nonpatient n5 affine FEBio solve

**Result, 2026-10-09: `passed_numerical_software_only`.** One released FEBio 4.13.0/Accelerate call solved the frozen `n5_affine` idealized 10 mm cube (1,331 nodes, 750 tet10 elements). The five saved states, four reported residual-converged steps, complete output parser, and analytic affine displacement, stress, deformation, energy, force-balance and signed-work checks passed. An independent read-only audit regenerated the deck and readout and matched the release, committed source, runtime bindings and saved output hashes. This is a numerical software check, not physical or patient validation.

| Recorded item | Result |
| --- | ---: |
| Native calls / retries / exit code | 1 / 0 / 0 |
| Supervisor wall time | 0.518976 s |
| Peak sampled process-group RSS | 49,561,600 B |
| Peak sampled active output | 2,145,877 B |
| Time-1 maximum affine displacement error | 7.806e-14 m |
| Time-1 maximum reconstructed stress error | 9.825e-7 Pa |
| Time-1 maximum deformation determinant error | 9.683e-11 |
| Time-1 integrated energy / signed-work errors | 2.609e-19 / 1.638e-18 J |

The supervisor enforced the frozen 600 s wall, 3 GiB sampled process-group RSS and 512 MiB active-output caps; none was approached. RSS and output sampling can miss brief between-sample peaks. These short-run measurements do not predict n9/n13 memory or patient-scale runtime. The receipt recorded output size before writing its final expanded JSON; the independently observed six-file directory afterward was 2,164,220 B, still well below the cap. The raw node, element and solver logs remain ignored locally.

An initial direct-file Python launch reported `ModuleNotFoundError: No module named 'scripts'` before an attempt directory or native call existed. Its stderr was not saved with the supervised attempt, so the independent audit treats it as a reported zero-native launch failure. The corrected `python -m scripts.mechanics_nonpatient_n5_supervisor` invocation used the same release and frozen source commit and produced the one audited native call. No retry of that native call occurred.

**Provenance.** Frozen source commit: `2f8f53f18111c2e839f677ac9c371e98f2f5175c`. Private one-call release SHA-256: `f2bebf5e1a7b722ff62e22603e63b80698e66fdc28fc3634ec078359f2726796`. Generated and saved deck SHA-256: `860dcdc8b29382281dd35ec41b6939bb29f1de5b3d3bd7a2e0adfd6f866a7674`. Saved receipt SHA-256: `9d26c0c57eefad06c53fe67b2bd64d9572778f23ed8b4356660654f88746f0ce`. The independent audit is retained locally at `build/nonpatient-n5-result-independent/REPORT.md`; its checks include the exact five saved non-receipt file hashes and a fresh parser/readout pass without another FEBio call.

The other six frozen benchmark cases, including nonuniform loading and n9/n13 levels, have not run here. Thus no mesh or time-step convergence, nonuniform response, patient-size feasibility, measured brain displacement, measured tool force, cutting interaction, injury model, route-planning reward, or clinical performance is established. The rejected Case4 mesh and patient-source gates remain closed.
