# One supervised HBE v5 N8 extended-endpoint compression solve

**Result, 2026-10-09: `passed_numerical_software_only`.** One FEBio/Accelerate call solved the frozen `compression:N8:S60:reference` specimen fixture from rest to the declared extended endpoint. All 61 saved frames, 60 converged steps, fixture constraints, sampled deformation, reaction balance, residual and work–energy checks passed. An independent read-only audit rehashed the release, 17 executing source files, 13 runtime files and five native outputs, then replayed the full stream readout from saved logs. This establishes a bounded numerical software result, not agreement with measured tissue force.

| Recorded item | Result |
| --- | ---: |
| Native calls / retries / exit | 1 / 0 / 0 |
| Supervised wall / peak sampled process-group RSS | 4.570366 s / 40,665,088 B |
| Peak sampled active output | 15,488,452 B |
| Final prescribed displacement / simulated signed force | -0.00073726 m / -0.0373150179847814 N |
| Final integrated reaction work / stored energy | 1.3005371342771914e-05 / 1.3005163640894274e-05 J |
| Maximum work–energy discrepancy / allowance | 2.077018776357165e-10 / 2.601856922954383e-07 J |
| Minimum logged / sampled reconstructed deformation `J` | 0.988549456468 / 0.789807300234 |

The largest normalized criterion was 0.000798284 against a ratio-1 threshold. Positive sampled `J` does not prove positivity everywhere in an element or physical material fidelity. The solver emitted two early `No force acting on the system` warnings during prescribed displacement; they are retained alongside the passing residual and reaction checks. The supervisor stayed below its 90 s wall, 3 GiB sampled RSS and 64 MiB active-output caps. Sampled peaks can miss short excursions.

**Provenance.** Frozen source commit `65c6d0b8b54514ae078bc72164a4d8dea416951c`; ignored release SHA-256 `417f34c234ed5c79eeaab4f9e060cd7e3a49b67acbe1de63730f018aa3afaad3`; executed deck SHA-256 `3cf4156b19191918841498490bc448f88a61b656b566d5d6a3d36db817d0a724`; saved receipt SHA-256 `8099c438caadf120d7269c14025fb92932fdda12a05f5ac8e8ade981cb20d45d`. Raw outputs remain ignored under `outputs/mechanics/hbe-v5-n8-one-shot-v1/attempt-01/`; the independent audit is local at `build/hbe-v5-n8-result-independent/REPORT.md`.

The other 11 frozen rows remain unrun. No measured HBE force curve was used, no force fit or mesh/time-step convergence was established, and the held-out torsion response remains closed. This postmortem specimen compression simulation does not validate patient retraction or cutting force, brain shift, neurological harm, or surgical decisions.
