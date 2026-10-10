# Goal-aware policy and distinct generated geometry splits

The new policy receives the existing spatial observation plus the public goal,
committed geometric probe contacts and explicit action modes. Its critic now
preserves which candidate geometry belongs to aspiration or probe; the initial
failed checks and source are retained. No legacy checkpoint is reinterpreted.
The shared planning adapter was independently executed with handwired test weights
and reproduced through the authoritative native replay. That is a software
integration check, not learned behavior.

A fixed generated family provides 24 distinct support geometries and candidate
column profiles, split before outcomes into 12 TRAIN, four SELECT and eight
held-out layouts. Each has two public goals. Geometry is visible through a
generated structural signal; private target labels are unused zeros. Family IDs
and roles are not neural features. Held-out task execution remains closed.
The earlier source-only recipe with fewer distinct candidate profiles is retained
in the proposal record; its revision preceded all task outcomes.

Root tests pass 22 policy and 36 structural-family controls. Independent reviews
pass 30 and 41 respectively. One separately declared positive factory/context
check executed only STOP on the first TRAIN and SELECT surface goals: six STOP
transitions and 90 geometry previews across sealing, direct execution and replay,
0.957 seconds total. It verifies integration, not useful family performance.

The next slice is the fixed TRAIN-only loss/update and checkpoint contract, then
one bounded matched IL/scratch-RL/SEARCH comparison on this same engine. No teacher
search, optimizer update, trained family result, patient transfer or physical
validation exists in this preparation. The current desktop offers the fixed
generated contact task; a learned contact selector awaits actual checked weights.
