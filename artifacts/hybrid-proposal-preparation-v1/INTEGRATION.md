# Fixed proposal-priority comparison preparation

The source validator now runs every existing admission guard before an optional
caller builds an action inventory. The default eager constructor delegates to
that validator and preserves its original construction/context logic. This
removes an otherwise hidden full-inventory cost from the proposed sparse pilot.
The canonical source-only/default suite passes **27 tests in 0.35 s**. Historical
staged old/new parity and full generated runner receipts are preserved here;
the phase-label failure in controls-03 remains a failed attempt.

The fixed one-fit experiment and its gates are in REPORT.txt. No acquired fit or
pilot outcome is claimed by this preparation commit. Runtime code is frozen
from local build sources after this commit; coefficients and patient arrays
remain local. TRAIN is 002/015/018/045; SELECT/EVAL remain closed.

Separately, the archived passive independent-control scripts now locate the
repository when run from their committed artifact directory. In a disposable
copy both the six-group review and two state regressions pass against the same
canonical source. The repair patch uses zero context and reconstructs the exact
before-source hash. Numerical code and original receipts are unchanged.
