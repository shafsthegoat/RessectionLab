# Expected policy behavior from the saved complete tree

No additional policy forward, gradient, simulator transition or patient read was performed. The calculation combines saved five-state action probabilities with all 16 nominal terminal-route returns. Every route association agrees with the 64 saved audited RL episodes within floating-point precision.

| Policy | Expected return | Any target reached |
|---|---:|---:|
| Initial | −0.275463 | 23.5496% |
| BC16 | −0.259146 | 16.5863% |
| RL16 | −0.270151 | 22.0374% |
| BC256 | +0.615612 | 63.1553% |

RL16 improved expected reward by only 0.00531247 while reducing target-reaching probability and worsening the greedy route. Thus “every measure worsened” would be incorrect, but useful planning improvement is not established. BC256 matches search under argmax while its stochastic policy remains imperfect. These percentages describe this finite simulated task, never clinical probabilities.

The fixed-order algebraic decomposition of RL's change is +0.00829694 from changed root probabilities with initial continuations, then −0.00298447 from changed continuations at final root probabilities. This is an arithmetic decomposition, not causal identification. Recomputing probabilities from float64 logits changes expected return by at most 6.34e−9. All original evidence remains unchanged.
