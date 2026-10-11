# Longer search stays negative and exposes a proposal omission

The declared 48-step comparison completes with **STOP and zero target progress for both methods**. Both plans seal before independent replay; exact complete histories and the saved independent audit pass. Both first-23-layer controls match the prior 24-step experiment. No time or transition cap is hit.

| Measure | Original search | Paged obstruction search |
| --- | ---: | ---: |
| Completed layers | 36, retained beam exhausted | 48, declared horizon |
| Expanded states | 70 | 95 |
| Evaluated native transitions | 805 | 1,950 |
| Search time, seconds | 116.480 | 330.593 |
| Full-cost previews, including setup and replay | 8,424 | 13,408 |
| Selected target / outside removal, mm³ | 0 / 0 | 0 / 0 |
| Selected return | 0 | 0 |
| Maximum explored opening depth, mm | 4.5 | 5.5 |

Doubling the horizon did not improve the useful-route outcome. Paged search stays at 5.5 mm opening depth from step 12 through 47, while its explored opening prefix widens from approximately 60 to 121 mm³ of outside-target removal. That negative prefix is not the selected route. All 4,940 emitted proximal/distal target proposals remain shaft-blocked; no target-directed action became feasible. All evaluated cumulative returns are negative.

The independent causal audit identifies a specific current-catalog omission. At the deepest step-47 prefix, actual blocker `[7,92,85]` is excluded because the optimistic existing-axis radial bound, 1.5811 mm, fits a 1.7 mm tool. That tool has no legal current action. Every legal action uses the 1.0 mm tool; none can remove the blocker. This is not evidence that a successful path was discarded, or that the missing centered movement is feasible. It shows why optimistic infinite-axis coverage is an invalid reason to suppress a finite blocker-centered candidate.

At that state, 16 of 35 observed first-blocker cells are excluded by that prerequisite and three more by the selection cap. Of 32 added candidates actually checked, 23 fail the aperture, six fail shaft clearance and only three are legal. The next targeted change is generic residual-blocker eligibility with conservative tool/aperture screening before bounded cell selection. Keep 48 steps, beam width, pages, world and rewards fixed; test the missing movement before a further full-route comparison. No horizon doubling or reward tuning follows from this negative.

The owned run takes **457.998 s**, sampled peak **544,047,104 B**, output **49,088,969 B**, **21,832 previews** and **2,759 total transitions**, with confirmed clean termination. All model loads, forwards, fits and optimizer updates are zero. This is one fixed-order geometric simulation on exposed SELECT013, not an RL gain, physical measurement agreement, neurological safety or global infeasibility proof. SELECT037 stays held and EVAL067 closed. No new data is acquired.

Executed source is commit `7cd2d61aaf49bcc89de12cd0e957b7cdab00efb6`. Exact result, parent receipt, release, runtime pins, controls, histories, independent certificates and compact causal witness are retained in the preservation index. Full inventory/page files stay local with a 440-file hash manifest; arrays and weights remain outside Git. Historical 24-step results and the initial generated-fixture failure are preserved.
