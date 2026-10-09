# One supervised n9 affine numerical-cube solve

**Result, 2026-10-09: `passed_numerical_software_only`.** The third frozen cube case completed one FEBio/Accelerate call on 6,859 nodes and 4,374 tet10 elements. All five saved states and four residual-converged steps passed the strict parser and analytic affine field, stress, energy and signed-work checks. Independent review replayed those checks from raw logs and verified release, committed source, runtime, deck and output identities.

| Recorded item | Result |
| --- | ---: |
| Native calls / retries / exit | 1 / 0 / 0 |
| Supervised wall / peak sampled process-group RSS | 3.649175 s / 275,103,744 B |
| Peak sampled active output | 10,748,550 B |
| Final maximum affine displacement error | 4.29203e-14 m |
| Final maximum reconstructed stress error | 1.90324e-6 Pa |
| Final stored energy / analytic energy | 5.3076762908706145e-6 / 5.307676290870663e-6 J |
| Final signed boundary work | -1.052416872519388e-5 J |
| Supplemental maximum free-node internal force, all five states | 7.813e-13 N |
| Supplemental maximum constrained-node signed force residual | 6.203e-13 N |

The independent supplemental check integrated internal forces at G8 points with the correct all-exterior affine boundary set. A top-centre finite-difference energy-gradient check agreed with the signed saved reaction to 2.588e-13 N. These are additional diagnostics; the original affine thresholds and verdict were unchanged. Four `No force acting on the system` warnings remain visible in the saved prescribed-displacement run.

The one call stayed within 600 s wall, 3 GiB sampled RSS and 512 MiB output caps. Sampled peaks may miss brief excursions. The frozen sequence has now consumed three of seven calls and 4.687578625 s of 1,800 aggregate native-wall seconds. Nonuniform n9, both n13 rows and the n13 half-step row remain unrun. The homogeneous affine case checks a known numerical solution and capacity; it does not establish nonuniform mesh/time convergence or tissue-law accuracy.

**Provenance.** Source commit `a61b82c9555ccb52bddba1927a27c9e0ce4eb8b9`; private release SHA-256 `e90840f8ad6afa08ecd9b3d722a7a87a6ba6f8ece7519633d1810f32115eb5bd`; deck `1dd6d1eeaa4072fed359f8217e1389a748bfafc065d230f643f607d4ff9645c9`; receipt `44775df0f7f04b2fdd6ec3f0ba43e595d34a07c6303142a5be1cb20d921f52b1`. Ignored raw outputs are under `outputs/mechanics/nonpatient-n9-affine-supervisor-v1/attempt-01/`; independent audit at `build/nonpatient-next-levels-gate/N9_NATIVE_RESULT_REVIEW.md`.

No patient geometry, measured tissue response, retraction/cutting force, neurological outcome, RL reward or clinical planning result was validated by this idealized cube experiment.
