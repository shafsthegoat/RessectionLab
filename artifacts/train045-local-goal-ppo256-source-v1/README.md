# Fixed 64-to-256 PPO continuation

Continue both audited seed0 checkpoints with exactly192 additional decisions each. Preserve all task assumptions, reward, inputs, architecture and PPO hyperparameters. Load and compare every saved parameter and Adam tensor/scalar exactly before the first update. Keep the inherited init/64 audits by hash; evaluate256 freshly from the complete start12 frontier and independently replay the full plan.

The continuation restarts the uniform genuine12–16 curriculum. It preserves optimizer state, not exact mid-episode physical/RNG continuity. Retire the previous unfinished episodes explicitly, union unique state/action identities, sum collected decisions and actual optimizer counts, and keep setup/evaluation/training cost segments separate. Existing input worlds, checkpoints and final patient partitions remain untouched.

Seven generated controls pass, including real standard-SB3 checkpoint restoration and continued optimizer updates. These controls do not establish learning success. The actual256 run was active at this source commit. Its caps are60minutes,4GiB RSS,140,000 native previews,1,000 transitions and128MiB output, within the existing192MiB shared local task reservation. One worker uses two CPU threads; no new acquisition or dependency is required.

Exact original paths and pins are in `INVENTORY.json`; qualification and pilot source/results remain in their earlier archives. Model weights, patient arrays and bulk traces stay local. The current retained decision baseline is the independently successful bounded search; the64 PPO endpoints fail. The next result must demonstrate better complete-task outcomes or equal outcomes at lower measured planning cost to change that decision.
