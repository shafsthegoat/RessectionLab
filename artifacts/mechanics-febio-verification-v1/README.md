# Prepared FEBio patch controls — not executed

Five fixed numerical decks and their hashes are in [decks/manifest.json](decks/manifest.json).
They use the tagged FEBio 4.13 Ogden alpha=2 law with explicit pressure model 1,
three-field hex8 domains, no augmentation, symmetric tangent and Skyline.
The 1000 Pa reference shear modulus is an arbitrary numerical unit check,
not a measured or fitted tissue property. Each 27-node cube has one free
interior node; all constrained components use prescribed displacements,
including zero values, to preserve signed reactions.

The owner analytical/XML/parser controls passed **40/40 in 0.51 s**. A separate
review added independent energy-derivative/objectivity, reaction virtual-work,
and rotated affine kinematics controls: the combined **43 tests passed in
0.50 s**. See [owner receipt](owner-tests-02.json) and
[independent review](independent-review.json). Earlier owner output is retained;
no failure was hidden. All positive output-parser fixtures were generated from
analytical expectations, not produced by FEBio.

No solver process has run and no specimen values or patient arrays were read.
Actual FEBio parsing, convergence and numerical agreement remain unverified.
The external runtime workflow must bind the executable/build, unchanged decks
and source, supervise the explicitly released runs, and retain every output
and failure. No sign, material parameter or acceptance tolerance may be fitted
to the resulting output. These patches do not establish specimen, patient,
mesh-convergence or near-incompressible locking validation.

[Preparation receipt](preparation.json), [tagged source receipts](source-receipt.json),
and [artifact index](artifact-index.json) preserve the prospective inputs.
See [the numerical contract](../../docs/mechanics-febio-verification.md) for
equations, units, reaction convention, primitive output fields and limits.
