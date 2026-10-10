# TRAIN-only diagnosis distinguishes imitation underfit from RL collapse

One fixed diagnostic reconstructed all 40 original TRAIN teacher states and compared initial, IL32 and RL32 checkpoints: 120 inference forwards, zero training updates or search, and no SELECT/MEASUREMENT outcomes or patient reads. It completed in 8.287 seconds at 280.3 MB sampled peak, with clean owned-worker termination. Independent saved-only arithmetic passes.

IL's fixed-state cross entropy improves slightly (1.86178 to 1.82265), but it greedily chooses STOP on all 40 states. The improvement comes from eight genuine STOP labels; loss on the 32 movement labels stays essentially unchanged. Among movement scores alone, teacher ranking improves from 1/32 to 22/32, including 16/16 probes. STOP still outranks every movement. This indicates some learned action discrimination plus a failed STOP comparison; it does not prove goal-specific spatial reasoning.

RL is more severely collapsed: mean STOP probability is 0.999990818 on these TRAIN states. No reversed-gradient or export defect was found. The next bounded question is whether a fixed full-teacher IL fit can learn this same boundary while retaining all STOP labels and existing loss semantics. More training compute must be reported honestly. These results do not authorize reuse of exposed measurement cases as an untouched test.
