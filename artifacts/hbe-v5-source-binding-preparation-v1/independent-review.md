# HBE v5 source-deck binding independent audit

**Decision: GO for the non-executable source/mesh identity and fixture-topology preparation only. NO GO for native execution, force/material validity, fitting, torsion access, patient mechanics, or clinical claims.**

## Exact reviewed source

| File | SHA256 |
|---|---|
| `manifests/experiments/hbe-01-03-v5-source-deck-bindings-v1.json` | `65ab3d9b2056e43e3a193b6d0288078f3617159483e271d0ab3cd0d624fb02e0` |
| `scripts/mechanics_hbe_v5_source_bindings.py` | `1ab6ee2c46aa93304efffdc88f3e4b6d57ade0a083d0907c3294708b6d093208` |
| `tests/test_mechanics_hbe_v5_source_bindings.py` | `91faee711c054dae82a076434ca1b3b98e94389314ae658e83f7f415afb492ed` |
| `docs/hbe-v5-source-binding-gate.md` | `705d167c4cb56446410da7dc08dd7c68e88457a3296add5609377405dd9e2f64` |

## Independent checks

- All 17 local ignored payloads existed and matched their pinned SHA256 bytes: ten old-domain FEBio source decks and seven representation-specific mesh JSONs. The ten decks mapped to all twelve ordered v5 rows; the two S120 rows reuse only S60 **source bytes**, with fresh schedule generation still a separate requirement.
- An independent Python/ElementTree audit rebuilt all six faces of every hex8, confirmed no face incidence above two, and found the listed bottom/top/side surfaces to be a one-to-one complete inventory of exterior faces on every mesh. All bottom nodes had the minimum z; all top nodes had the maximum z. Full native height was 0.00489159 m; reconstructed lower-half height was 0.002445795 m.
- For all ten decks, independently compared source NodeSet bottom/top membership to mesh JSON, found one zero and one axial controller, and checked every prescribed displacement. Full physical models fix both plates x/y/z, using zero x/y and axial z on each singleton top node. Lower-half models fix bottom x/y/z and prescribe only half-factor axial z at every artificial-midplane top node, leaving tangential DOFs free. Side-rim nodes overlap plate sets where geometrically appropriate; there is no independent side BC.
- Focused synthetic and local-source tests: `15 passed in 1.35s` using `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_mechanics_hbe_v5_source_bindings.py`. The full local binding validator returned ten checked decks, seven meshes, twelve rows. Tests guard missing/extra/swap/topology/BC defects, and `require_execution_ready()` always raises. No native solver or mesher was launched.
- This gate reads already tracked coordinate-only diagnostic metadata through v5/v4 preparation; it does not open raw measured axial files, measured response values, held-out torsion payloads, patient files, or native logs. It does **not** establish physical accuracy or authorize a read of those later data.

## Bound mesh identities

| Mesh | SHA256 | Nodes | Hex8 | Exterior faces |
|---|---|---:|---:|---:|
| `full:N8` | `3a6b5bb720bd3998f035abecf15d5c603d46c5ed281ce1d08c4aadec69c1dab9` | 1045 | 768 | 512 |
| `full:N12` | `0ed154be0d17c8baaac6ee0288381a53ec100f93cd936a194202f69ef32ae033` | 3199 | 2592 | 1152 |
| `full:N16` | `5fefdc7e0ba068461ea5d64074e11fe734a6460552798487fb2794ff65075f6b` | 7209 | 6144 | 2048 |
| `half:N16` | `c1665d5f107c5e20590d877e7613910a5a2aef1c54c9dff5f5229a8d4f8927ee` | 4005 | 3072 | 1792 |
| `half:N24` | `f8a934c3e035fbf478583719b901490f72478579b72d0b4d8b0a6dea1a3b99a2` | 12439 | 10368 | 4032 |
| `half:N32` | `edbb95dd61e2f1ba94faf405bbd263a2d995dcc87ebd298a8e072595ecc6009a` | 28233 | 24576 | 7168 |
| `half:N36` | `8e7ad0e636151e007ab9fd8af0d043abfcf889b960c53e726766126691c3c075` | 39610 | 34992 | 9072 |

## Bound deck identities

| Deck | SHA256 | Mesh |
|---|---|---|
| `compression:N8` | `b0191042df6fccc4c4dc317bb9145f0a077880a7e5589cd471b9ac93452a2506` | `full:N8` |
| `compression:N12` | `0860b181fc427979b463f516d83fdbd10e9357b90390138f2b2b6c2212101440` | `full:N12` |
| `compression:N16` | `706a323f0bc0a6b34264b34c4f137529f4583aff7375416286925c773005ef7b` | `full:N16` |
| `compression:N24` | `d6b87f944d3cc2f9ab1d5849a806cfa762f6d810ddde88308ec8c9906fd98706` | `half:N24` |
| `compression:N32` | `f481b4bb872ad1982eddd16544e579bef3caf96bbf1a91a2e2ad01f9446740e3` | `half:N32` |
| `compression:N36` | `93e02fff10bb680c7f60b93161a217bd47146b4d0dba34316003510df1008847` | `half:N36` |
| `tension:N8` | `474c1f0fdfa5769ab53c0dd72933e63961a90da4db922e57e8b51695b0149a5a` | `full:N8` |
| `tension:N12` | `b3cbf26b826b700aab89273f13324852690be09947d087acff6bebd92b218294` | `full:N12` |
| `tension:N16` | `43079eb759809ffffc16c4df329c73b23fc3ff4c8f985838397736162d2b9982` | `half:N16` |
| `tension:N24` | `0b66988b46c7d9af1286b24a646ba6279975f17c9bda9d3c78a7b242bdc670c5` | `half:N24` |

## Twelve-row mapping

| Run | Source deck |
|---|---|
| `compression:N8:S60:reference` | `compression:N8` |
| `compression:N12:S60:reference` | `compression:N12` |
| `compression:N16:S60:reference` | `compression:N16` |
| `compression:N24:S60:reference` | `compression:N24` |
| `compression:N32:S60:reference` | `compression:N32` |
| `compression:N36:S60:reference` | `compression:N36` |
| `compression:N36:S120:reference` | `compression:N36` |
| `tension:N8:S60:reference` | `tension:N8` |
| `tension:N12:S60:reference` | `tension:N12` |
| `tension:N16:S60:reference` | `tension:N16` |
| `tension:N24:S60:reference` | `tension:N24` |
| `tension:N24:S120:reference` | `tension:N24` |

## Remaining dependency

The source-deck hashes here are local selected old-domain payload identities, not independent proof of material law, complete control/solver primitives, correct new-domain adapted schedules, numerical convergence, or physical force agreement. A separately reviewed, bounded release must authenticate the adapted decks and native output before any fitting or held-out validation. The full/half fixture difference is an explicit modeling assumption, not a measured equivalence.
