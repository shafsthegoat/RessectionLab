# Independent axis inspection review

All 33 independent tests pass in 0.92 seconds on facade source
`ffb7fa0283194711ef24730dfe088c6d2ddb9108e10537c04d32474ef1ed05a1`.
The source stayed unchanged during that run. Ten existing geometry, source,
support, simulation and selected-route modules match the committed baseline;
this slice adds a facade, tests and documentation only.

Two failures are retained with the earlier facade source:

- The final cancellation callback could change the source case ID while cached
  case/planning hashes stayed unchanged. The initial run had one failure and
  22 passes. Fresh full-source identity checks now reject this at entry or exit.
- One valid oblique access succeeded as RAS but its equivalent LPS source was
  refused because repeated unit-vector normalization changed a component by
  1.11e-16. The retained reproduction fails in 0.09 seconds. A separate bounded
  256-vector algebra probe found 75 such differences; that count describes its
  chosen algebra inputs, not patient anatomy or clinical prevalence.

The repaired LPS boundary records requested and actual directions and permits
only four float64 epsilons of absolute component roundoff, with zero relative
tolerance. Centers, radii and window IDs remain exact. Eight factory-drift
adversaries still reject meaningful changes, and actual nonaxial requests remain
unsupported. Candidate entry, endpoint and axis outcomes agree in the named
tiny RAS/LPS fixtures; native configuration bytes and provenance are not claimed
identical across source conventions.

Other checks cover source support prohibitions, pending reviews, separate
acknowledgments, complete blocked inventories, detached results, cancellation
after a preview and late source/tool/reward/world mutation. A supported complete
inventory with no legal cut retains its full omission/rejection ledger; it
grants neither candidate eligibility nor tissue-removal authority.

`initial-tests.txt`, `lps-negative-tests.txt`, `reproduce-lps-negative.py` and
`pre-review-source.py` preserve the failures. `repaired-tests.txt` and
`review.json` bind the passing result and reviewed files. No public patient,
training, bridge or UI work ran in this review, and no geometry model changed.
