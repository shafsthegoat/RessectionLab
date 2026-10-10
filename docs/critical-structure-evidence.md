# Critical structure evidence and planner integration

The selected case now owns a typed registry for motor, language and vessel
annotations. This is a source-consistency contract and planner connection.
It does not authenticate arbitrary imported source declarations or establish
vascular accuracy. No eligible positive image–vessel-annotation pair has yet
passed acquisition and admission; the corresponding real-case gate remains open.

## Input contract

`CriticalStructureEvidence` binds one selected annotation per structure to the
case ID, original acquisition SHA256, current image-array hash, exact source
grid and canonical RAS+ millimeter affine. Source-derived dataset, participant
and timepoint identifiers must agree. Linkage, annotation domain, derivation,
rights, lineage and human review each retain their own source references.
Only identity-grid derivation is currently supported. Registration or resampling
needs a separately admitted transform; matching array dimensions is insufficient.

The Boolean annotation domain is separate from positive labels. Labels outside
it are invalid; an empty domain is excluded. Domain coverage means where a source
claims annotation was performed, not complete vessel detection. Missing, partial,
unreviewed and lineage-ineligible evidence remain distinguishable. A reviewed
label is still not a claim that unseen perforators are absent. These records
cannot become clinical deficit probabilities or training trajectories.

Review binds the exact annotation, domain and provenance content. Annotation
and review availability must both precede a timestamped planning cutoff;
unknown dates are excluded there. Without a cutoff, the receipt explicitly
describes retrospective image geometry with availability unassessed. File
modification dates do not supply clinical availability. Learned ancestry goes
through the real-observation policy guard; manual correction cannot erase it.

Arrays have immutable byte storage and checked shape/dtype/layout. Save/reopen
retains separate annotation and domain arrays. Full case identity includes the
selected records and invalidates old reports when evidence changes. Planner
bindings omit excluded records; absence and an entirely excluded registry share
one empty binding. Full-history hashes still invalidate artifact identifiers.
Both beam searches now break equal-score ties by stable proposal traversal,
so those identifiers cannot reorder equally scored branches.

## Actual consumers

| Consumer | Implemented behavior | Limit |
|---|---|---|
| Static route search / both desktop route operations | Resolve case-owned masks; supplied vessel labels are always hard exclusions, including custom search configurations; retain full-tool swept annotation-domain counts | Partial maps do not enter Pareto contact objectives; outside-image anatomy remains unknown |
| Native factories and axis inspection | Resolve exact canonical vessel exclusions and bind their provenance | Motor/language records in this new registry are explicitly unsupported here; existing FunctionalEvidence is a separate input |
| Coarse geometric simulator | Max-pool positive vessel labels; any labeled source cell excludes its whole coarse block | Conservative geometric occupancy, not measured interaction or vascular injury |
| Independent native history audit | Resolve canonical exclusions for CaseData; reject caller substitutions | Lightweight geometry-only objects remain a separate internal API |
| Native replay / desktop bindings | Compare current eligible receipt and mask hash/provenance, and require a binding when eligible critical evidence exists | Self-consistent saved hashes alone do not prove consumption or clinical validity |
| Historical spatial-learning adapter | Refuse eligible critical records it cannot consume | No current compliant learned-planning result |

Raw replacement masks cannot bypass the registry. The desktop exposes coverage
and exclusions through case descriptors and `inspectEvidence`; this milestone
does not add a renderer annotation editor or positive vessel demonstration.

## Verification and unresolved acceptance

```sh
.venv/bin/python -m pytest -q \
  tests/test_critical_evidence_real_source.py tests/test_real_observation_policy.py
```

The completed run passed **64 checks in 4.95 seconds**: 22 source/metadata/clock
and ordering checks plus 42 existing policy refusal checks. Actual anatomy is
the hash-verified original Lausanne TRAIN sub-000 T1. Its MRI and hashes survive
save/reopen; the desktop reports absent anatomy as missing and refuses route
generation without required brain support. Protocol refusals, timestamp
comparisons and scalar sort identities create no patient annotation, event or
simulator trajectory. The source-dependent checks skip if that original is not
locally acquired. TypeScript checking passed with the bundled Node runtime;
the earlier default-PATH attempt failed because `node` was unavailable.

Independent source reviews exposed and prompted repairs to review-time filtering,
coordinate truncation/wrapping, array-layout mutation detection, optional-mask
bypasses, empty coverage, unconsumed native functional records, replay binding,
coarse/audit consumers and hash-dependent beam ties. Reviews found no further
blocker to this bounded contract milestone. They did not run positive vascular
episodes or authenticate source annotation contents.

Still required: acquire and independently verify an eligible same-person
image/annotation pair, source review and domain evidence; test a real route
alteration/refusal through the desktop/API, positive save/reopen and stale-result
behavior; quantify annotation errors against independent real references.
The current missing-evidence checks cannot prove those outcomes.

### Generated consumer follow-up under later human steering

The later steering permits generated software controls. The [three positive
bridge controls](../artifacts/generated-critical-consumer-controls-v1/ROOT_REVIEW.md)
now prove that both actual route APIs consume resolved vessel exclusions and
reject stale route/run bindings. They keep requested geometry unchanged and
inject only the typed resolver result; real-source admission remains untouched.
All fixture provenance is explicitly simulated, with no fictional human review.
These controls pass in canonical tests but do not close the real-pair, rendering
or compatible learned-planning requirements above.
