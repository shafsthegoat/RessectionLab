This read-only refresh covers the saved primary at **446 actions** (still
running) and the completed **73-action vascular alternative**. The detailed
[JSON inventory](tool-family-inventory-current-20261020T123831Z.json) records
exact native hashes, state IDs, 1-based action positions, source counts, prior
fixture receipts, and matching vascular telemetry evidence. No physics replay,
GUI check, production edit, or clinical outcome assessment was performed.

| Tool family | Patient evidence in these saved branches | Engineering fixture |
|---|---|---|
| Cranial drill | Primary endpoint: 3,919 mm³ owned bone removed. Vascular actions 2–4, 12, 24, 30–31, 37–38 have observed drill-work events totaling 3,453 mm³. | Existing explicit fixture passed |
| CUSA | Primary endpoint: 11,595 mm³ owned target removed; 31,978 mm³ retained. Vascular removal events total 312 mm³. Exact action lists are in JSON. | Existing explicit fixture passed |
| Frazier suction | Primary activation at 69–70, followed by stationary shared-clock exposure; endpoint debris removal 177.103 mm³. Vascular actions 68–69 remove 5.966 mm³ fluid. | Existing explicit fixture passed |
| Micro scissors | Vascular action 48: five transection events; endpoint five nonpatent damaged faces in one wound. | Existing explicit fixture passed |
| Bipolar forceps | Vascular actions 55–56: observed energy; all five faces end at actual coagulation dose 2.0 and clot 1.0, with no active wound. | Existing explicit fixture passed |
| Irrigation | Vascular actions 62–63: 6 mm³ observed irrigation. Modeled blood emission is recorded separately. | Existing explicit fixture passed |
| Brain retractor | No action/effect in these branches. Supplied soft tissue exists; patient retraction remains unproved here. | Existing explicit fixture passed |
| Scalpel | No action/effect in these branches. | Existing explicit fixture passed |
| Periosteal elevator | No action/effect; periosteal attachment evidence absent. | Existing explicit fixture passed |
| Craniotome | No action/effect; no supplied anchors establish flap separability. | Existing explicit fixture passed |
| Dural scissors | No action/effect; dura/membrane anatomy absent. | Existing explicit fixture passed |
| Dissector | No action/effect; soft/tumor attachment evidence absent. | Existing explicit fixture passed |
| Tumor forceps | No action/effect or free-fragment extraction proof; attachments unknown. | Existing explicit fixture passed |
| Cortical probe | No action/effect or supplied functional response evidence. | Existing explicit fixture passed |
| Subcortical probe | No action/effect or supplied functional response evidence. | Existing explicit fixture passed |

The matching prior vascular telemetry contains **63 accepted and 10 partial
committed actions**. Four partials are drilling progress and six are material
contact stops. These are not described as entirely accepted. Rejected proposals
are stored separately; rejected attempts cannot be recovered from the native
accepted/partial action log alone.

Owned non-target soft-tissue removal is zero at these recorded endpoints.
That is an exclusive material-ownership count: the primary removed source cells
also have **11,779 memberships in the raw unreviewed brain annotation**. Raw
brain/target overlap is not clinical healthy-tissue truth and does not change
the 43,573 mm³ owned-target denominator.

The earlier `vascular/tool-family-inventory.json` remains unchanged. Prior
engineering fixtures supply explicit synthetic anatomy/properties, including
fluid/attachment/functional declarations absent from the patient. No patient
dura, attachments, complete vessel network, functional responses, measured
forces, injury, expertise, safety, or clinical outcomes are inferred.
