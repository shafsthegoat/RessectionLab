# HBE v5 reference-source binding preparation

The [binding declaration](../../manifests/experiments/hbe-01-03-v5-source-deck-bindings-v1.json)
maps twelve proposed HBE_01_03 reference runs to ten exact old-domain decks
and seven exact specimen meshes. The [validator](../../scripts/mechanics_hbe_v5_source_bindings.py)
checks their hashes, every mesh node and hex8 element, complete exterior faces,
and the top/bottom fixture constraints. Full specimens use bonded physical
plates; lower-half models use a free-tangential artificial midplane with half
the axial displacement. S120 rows reuse only source bytes and still require
fresh full-schedule adaptation and new native solves.

All ten local deck/mesh pairs passed source/fixture checks. The writer and
root each passed 43 focused and existing v5 tests. The
[independent audit](independent-review.md) matched all 17 local payload hashes,
reconstructed boundary topology and passed 15 focused controls. This is a
non-executable preparation: source decks have not been adapted/saved for v5,
FEBio has not run, and no new force, fit, convergence or held-out result exists.
The next gate is new-endpoint native primitive/output validation and a bounded
twelve-row comparator before any specimen physical qualification. Nothing here
validates patient retraction, cutting, injury or material properties.
