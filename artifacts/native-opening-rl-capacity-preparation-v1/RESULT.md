# Fixed longer scratch RL preparation

The single declared run extends the original scratch RL learner to 256 updates while preserving architecture, initial seed, optimizer, reward, entropy/value terms, four fresh on-policy episodes per update and two-action horizon. Its first 16 sampled trajectories, losses and gradient/hash chain must reproduce the original exactly. The fixed final checkpoint remains primary; no intermediate result selects or extends training.

Six no-gradient readouts distinguish expected reward, any/full target reaching, STOP probability and deterministic behavior. The complete saved nominal tree is used only for assessment. It supplies no teacher initialization, cached training trajectory or loss label. All 1,024 training episodes receive the existing independent execution audit before optimization.

Owner validation passed 16 focused controls, including exact reproduction of the first four-episode update. Independent review passed 27 targeted controls without actual model execution, including altered privileged evaluation returns that leave the loss inputs unchanged. It found no blocker for one 180-second, 2-GiB, one-thread attempt. Actual resource measurements and any failure must be retained.

This remains an optimization diagnostic on one generated task. It cannot establish patient generalization, physically valid surgical performance or superiority over search. The earlier negative RL result and positive BC capacity result remain unchanged.
