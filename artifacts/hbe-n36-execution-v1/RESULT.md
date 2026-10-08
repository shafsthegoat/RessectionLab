# N36 conditional consistency screen passed

The single declared FEBio solve completed without retry. Independent replay of its saved native primitives reproduced the full readout, complete comparison and all 240 signed triplet estimates. All 60 nonrest states pass the declared conditional consistency screen, with no exceedance, order-drift or limit-instability flags.

| Quantity | Observed result |
|---|---:|
| Finest adjacent force change, N32 to N36 | 0.049000 mN |
| Conditional two-limit remaining envelope | 0.642702 mN |
| Declared force allowance | 0.723189 mN |
| Endpoint fitted convergence order | 0.700551 |
| Endpoint native reaction | −36.151426 mN |
| Native solve time | 873.3347 s |
| Full phase through result publication | 1,080.2427 s |
| Sampled peak process-group RSS | 1,827,356,672 bytes |
| Retained output including closeout | 725,949,797 bytes |

The earlier N32 envelope failure is retained. This is a **candidate for temporal review**, not accepted spatial convergence or a rigorous continuum error bound. Calibration remains closed; physical validation is unmeasured. The smallest observed endpoint perturbation that changes the conditional classification is about 0.8317 µN. Therefore passing the original 0.2% step-size tolerance alone would not establish classification stability: the next prospective load-step check must report its actual observed shifts against that much smaller sensitivity.

Exactly one native solve and no new mesh generation occurred in this phase. No measured force curve was accessed. The independent review took 90.01 seconds and is separate from simulation runtime. Prior accepted replay records and all source/input/output bindings are unchanged. No automatic further uniform refinement, calibration or new solve follows from this result.
