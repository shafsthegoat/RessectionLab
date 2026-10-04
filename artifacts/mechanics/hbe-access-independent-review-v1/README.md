HBE access review: repairs pending
=================================

Six disclosure/provenance failures were reproduced against access source6405e074 using temporary constructed ZIP archives and fake receipts; changed archive bytes correctly rejected. Exact original source, helper fixtures, test versions and negative output are retained. The initial run also encountered stale helper metadata; that fixture-only correction is explained in `negative-check.json` and is not represented as a production fix.

Current findings and pending status are in `findings-pending-repair.json`. No real HBE curves, patient values, image data or FEM runs were accessed. This review is not release approval.
