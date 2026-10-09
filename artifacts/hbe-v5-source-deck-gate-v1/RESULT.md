# HBE_01_03 v5 source-deck execution-gate audit

Read-only audit, 2026-10-09 UTC. **Disposition: source candidates and their mesh/boundary topology are locally recoverable; NO-GO for native execution or numerical/physical qualification.** No FEBio solver, mesher, measured response, held-out torsion, patient image, or fit was opened or run. The only computation was file hashing, XML/JSON structural comparison, and the already committed pure v5 deck adapter in memory. No tracked file was edited.

## Exact available old-domain source candidates

Paths below are relative to the repository root. The v5 manifest still has `source_deck_sha256: null` for every row, so these are **audit candidates**, not approved v5 release bindings. S120 variants reuse the corresponding S60 old-domain source and regenerate the entire new-domain time/load schedule; they do not reuse an old numerical solution.

| V5 branch/mesh | Native domain | Old Skyline source deck | Local SHA256 |
|---|---|---|---|
| compression N8 | full | `outputs/mechanics/hbe-01-03-poc-v1/mesh-preparation/N8/generated/compression-60-reference/specimen.feb` | `b0191042df6fccc4c4dc317bb9145f0a077880a7e5589cd471b9ac93452a2506` |
| compression N12 | full | `outputs/mechanics/hbe-01-03-poc-v1/mesh-preparation/N12/generated/compression-60-reference/specimen.feb` | `0860b181fc427979b463f516d83fdbd10e9357b90390138f2b2b6c2212101440` |
| compression N16 | full | `outputs/mechanics/hbe-01-03-resolution-v1/mesh-preparation/N16/generated/compression-60-reference/specimen.feb` | `706a323f0bc0a6b34264b34c4f137529f4583aff7375416286925c773005ef7b` |
| compression N24 | lower half | `outputs/mechanics/hbe-01-03-halfheight-spatial-v1/preparation/cases/compression-N24-S60-reference/skyline.feb` | `d6b87f944d3cc2f9ab1d5849a806cfa762f6d810ddde88308ec8c9906fd98706` |
| compression N32 | lower half | `outputs/mechanics/hbe-01-03-halfheight-global-n32-v1/preparation/cases/compression-N32-S60-reference/skyline.feb` | `f481b4bb872ad1982eddd16544e579bef3caf96bbf1a91a2e2ad01f9446740e3` |
| compression N36 | lower half | `outputs/mechanics/hbe-01-03-halfheight-global-n36-v1/preparation/cases/compression-N36-S60-reference/skyline.feb` | `93e02fff10bb680c7f60b93161a217bd47146b4d0dba34316003510df1008847` |
| tension N8 | full | `outputs/mechanics/hbe-01-03-poc-v1/mesh-preparation/N8/generated/tension-60-reference/specimen.feb` | `474c1f0fdfa5769ab53c0dd72933e63961a90da4db922e57e8b51695b0149a5a` |
| tension N12 | full | `outputs/mechanics/hbe-01-03-poc-v1/mesh-preparation/N12/generated/tension-60-reference/specimen.feb` | `b3cbf26b826b700aab89273f13324852690be09947d087acff6bebd92b218294` |
| tension N16 | lower half | `outputs/mechanics/hbe-01-03-halfheight-spatial-v1/preparation/cases/tension-N16-S60-reference/skyline.feb` | `43079eb759809ffffc16c4df329c73b23fc3ff4c8f985838397736162d2b9982` |
| tension N24 | lower half | `outputs/mechanics/hbe-01-03-halfheight-spatial-v1/preparation/cases/tension-N24-S60-reference/skyline.feb` | `0b66988b46c7d9af1286b24a646ba6279975f17c9bda9d3c78a7b242bdc670c5` |

N8/N12 full source hashes are already in `artifacts/mechanics/hbe-01-03-mesh-preparation-v1/deck-index.json`; selected later hashes also occur in prior preparation/execution receipts. Their historical presence does not itself release them for a new experiment. The v5 manifest's twelve rows are ten unique source decks plus `compression:N36:S120` and `tension:N24:S120` adaptations.

## Independent topology and adapter checks

For all ten old-domain candidates I recomputed the source and mesh SHA256, parsed the actual XML, and compared every XML node ID/coordinate, hex8 element ID/connectivity, and bottom/top/side quad surface against the corresponding local `mesh.json` arrays. All ten had exact equality, and each mesh hash equalled the representation declared by v5. The full-domain node/element counts were N8 1,045/768, N12 3,199/2,592, N16 7,209/6,144; half-domain counts were N16 4,005/3,072, N24 12,439/10,368, N32 28,233/24,576, N36 39,610/34,992.

For each candidate I compared complete top and bottom NodeSet memberships with the hash-bound mesh boundary lists; every top singleton NodeSet contained exactly its named node; every declared top node had exactly the required prescribed-displacement DOFs; the physical bottom had exactly x/y/z and no side BC. Full-native top plates had x/y/z on every top node, and lower-half artificial midplanes had only z with factor 0.5 and free x/y. Counts of top nodes were N8 209, N12 457, N16 801, N24 1,777, N32 3,137 and N36 3,961. No candidate had a missing whole top-node BC, the defect the v5 fixture checker alone could miss. This is local source/mesh topology consistency, not solver convergence or physical fidelity.

`v5.validate_preparation()` passed. `v5.adapt_deck(...)` accepted all 12 proposed run IDs against these actual local old-domain bytes, including both S120 schedules, and its delta/backend checks returned pure prepared XML in memory. No generated deck was saved and no native output was produced. `outputs/mechanics/hbe-01-03-branch-calibration-v4` and `...-v5` were absent at audit time. The v5 adapter's receipt deliberately says `source_deck_identity_approved_for_execution=false`.

## Published boundary-condition meaning

The original HBE study fixed cylindrical human postmortem specimens to sandpaper-covered upper and lower holders with superglue. Its glued condition fixes the upper/lower surfaces in-plane, unlike an ideal slipping plate, and the paper reports that the glued/slipping choice strongly changes nominal stress. The repository's full-native axial decks encode bonded top/bottom and traction-free curved sides. Their lower-half top is an **artificial symmetry midplane**, not a second physical glued plate: only axial displacement is constrained there and in-plane motion remains free. This distinction must survive native primitive checks and reconstruction. Published specimen fixture data support testing force under that fixture; they do not supply live patient retractor contact, cutting forces, or patient-specific material parameters. Primary source: [Hinrichsen et al., *Biomechanics and Modeling in Mechanobiology* (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10511383/); repository interpretation: `docs/tissue-mechanics-measurements.md` and `docs/hbe-halfheight-equivalence.md`.

## Nearest actual release blocker

The first concrete release step is a separate immutable pre-execution manifest that binds **these ten exact source deck bytes and seven representation-specific mesh bytes** to the twelve rows, reruns the full source/mesh/BC-topology checker from an archived commit, and receives independent review. The v5 manifest currently rejects execution; neither the local inventory above nor fixture tests silently change that.

The next essential implementation gap is an **extended-endpoint native primitive reader/checker and twelve-row qualification comparator**. Existing old-domain solver logs cannot establish any new-domain 61/121-frame displacement, reaction, Jacobian, balance, energy/work, 75-probe, reconstruction, or spatial/temporal convergence result. A supervised one-shot runner, bounded resource caps/output budget, independent publication replay, and explicit no-fit-on-failure gate must be frozen before solving. The v5 source review and v4 proposal already require these; no existing output can substitute. V3 remains terminal no-fit after 29/30 axial-coordinate coverage in each branch, and the earlier global spatial failure remains. These blockers are software/evidence release work, not a missing set of source-deck files.

Even twelve successful reference solves would only qualify a processed **one-specimen** axial numerical/calibration foundation. Later response fitting and withheld same-specimen torsion are separate gates. None of this alone validates patient retraction, tissue injury, force under a surgical tool, or a planning policy.
