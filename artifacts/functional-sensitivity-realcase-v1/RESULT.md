# Real-patient whole-tool functional sensitivity

The declared analysis completed on **one previously consulted UCSF-PDGM-0004
case**. Two unchanged native tool histories passed fresh independent geometry
checks. Their modeled motor-map encounters and surrogate costs respond to tool
geometry and nonzero registration uncertainty. This is population-prior
sensitivity, **not patient-specific function or postoperative deficit risk**.
No training, policy updates, new route search or outcome-driven configuration
changes occurred.

The executable alternatives use the same previously defined axis with two
different existing instruments. Three other existing rays remain rejected for
native execution because their shafts encountered remaining tissue; their
functional event lists are empty in every panel. Their separately passed static
access checks do not establish executable removal. The original 54 rejection
records remain in `analysis/geometry.json`. STOP remains a separate control.

| Frozen registration scenario | Tool | Motor supplied-map event | Conditional 95% MC interval | Mean motor surrogate | Worst 5% motor surrogate |
| --- | --- | ---: | --- | ---: | ---: |
| Gaussian SD 1 mm / 1° | Fine | 51 / 64 | [0.683, 0.877] | 50.01 | 105.09 |
| Gaussian SD 1 mm / 1° | Wide | 64 / 64 | [0.943, 1.000] | 146.12 | 280.79 |
| Gaussian SD 2 mm / 2° | Fine | 43 / 64 | [0.550, 0.774] | 54.74 | 133.71 |
| Gaussian SD 2 mm / 2° | Wide | 53 / 64 | [0.718, 0.901] | 153.91 | 357.72 |

The event is any complete-tool-contacted source tissue cell whose covered
resampled map value is **at least 0.5**. The threshold is an operator-defined
support sensitivity choice. Intervals describe finite Monte Carlo error
conditional on this fixed model; they do not cover model validity or clinical
uncertainty. Surrogate units are unitless map support × touched-cell mm³, not
removed tissue or a sum of clinical probabilities. Worst 5% is empirical
upper-tail CVaR with fractional boundary mass.

Broader uncertainty lowered these binary motor event frequencies while raising
mean and tail exposure costs. A lower event count alone therefore does not imply
a better plan. Language encounters were 64/64 for both tools in both nonzero
scenarios; mean language surrogate changed from 100.73 to 98.42 for fine and
290.86 to 284.46 for wide, while worst-tail costs increased. No ranking reversal
was required or manufactured.

The prespecified 0.1 support threshold produced motor and language events in
64/64 worlds for both tools. It changed the event definition, not the underlying
continuous costs. The two nonzero generators each produced 64 numerically
distinct transforms. Each 32-seed deterministic control produced exactly one
transform and no Monte Carlo interval; both tools encountered both maps in
32/32 deterministic replays. These are repeated simulations within one patient.

With both maps supplied, these two tool footprints had full released-map
sampling coverage in the realized scenarios. This does **not** establish
individual functional coverage. In the motor-only/language-only controls the
omitted component produced 64 unknown worlds, null frequency and null total
surrogate, with 189 mm³ of unassessed contacted cells for fine and 543 mm³ for
wide. STOP had no contact, including when a component was missing.

| Fixed native history | Simulated target removed | Simulated normal removed | Residual target | Full-tool touched-cell surrogate |
| --- | ---: | ---: | ---: | ---: |
| Fine | 19 mm³ | 1 mm³ | 41,900 mm³ | 189 mm³ |
| Wide | 174 mm³ | 11 mm³ | 41,745 mm³ | 543 mm³ |

Removal counts use only independently certified fully contained connected source
cells. Complete-tool contact includes retained and partially contacted tissue;
it is not additional removal. These short histories address a small route slice,
not a complete resection strategy.

The parent measured **28.984 s**, the worker **28.242 s** and worker peak RSS
**2.108 GiB**, within the unchanged 180 s / 3 GiB limits. No retry was needed.
Runtime versions, source hashes, all eight panel freezes and raw outcomes are
saved under `analysis/`; original data and source bytes were unchanged.

| Measured worker phase | Seconds |
| --- | ---: |
| Verify and load pinned inputs | 1.770 |
| Independent geometry and contact footprints | 10.966 |
| Evidence versions and all panel freezes | 7.063 |
| New evidence case save/reopen | 5.914 |
| Eight event-evaluation panels combined | 2.339 |

The main avoidable work was preparing footprints for the three already rejected
rays: 6.606 s. The two valid native footprints took 0.121 s together, and their
fresh independent audits took 3.109 s. This identifies a possible later lazy
evaluation improvement; no speedup has been executed or claimed. Ordinary
desktop activity and another agent's structural regression checks were not
controlled, so these are workflow cost receipts, not isolated-machine latency
benchmarks.

Execution used commit `f3582cc05f24d41407f70063855df26722e031d2`, archive
`75f43247d37c39224689b166b101f9df05d626f2e91e249281fa85128adca470`, and
declaration `b904b340fc47e488403312da7f222568756eb39a547db5ea84d762c73d10164d`.
The new primary Gaussian evidence bundle is separate from the original case and
the earlier uniform roundtrip fixture. Its hash is recorded in the machine
report; medical arrays are not included in Git.

The structural mirror's official byte equivalence remains unverified. Atlas
alignment, brain support and hypothetical access remain unreviewed; the global
alignment does not resolve tumor mass effect. Patient diffusion, vascular
coverage, tissue forces and clinical outcome calibration remain unavailable.
The result does not establish RL superiority, population pretraining,
new-patient generalization or clinical use.
