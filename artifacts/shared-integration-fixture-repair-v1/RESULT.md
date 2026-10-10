# Restore meaningful native and learning regression checks

Five failures reproduced on the prior committed baseline came from fixtures that
predated current case/context contracts. The tests now explicitly declare their
generated source and horizon, and the native case stub includes missing context.
Production source and admission guards are unchanged.

The repaired canonical files pass **34 tests in 1.37 seconds**. Three added negative
checks retain missing-context refusal. Existing invalid-experience tests now reach
their intended stale-action, masked-action, incomplete-episode and nonfinite-reward
checks, instead of accidentally passing at the outer missing-context guard.
Analytical policy gradients retain the same expected values.

These are tiny unit controls, including one ephemeral optimizer update. They are
not a new training experiment or scientific performance evidence. No patient or
checkpoint was read. The original failure evidence remains in the shared episode
integration artifact; no whole-repository green status is claimed.
