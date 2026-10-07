# Medivis patent concepts and RessectionLab project opportunities

October 6, 2026

The strongest portfolio direction is **multitool planning that accounts for what the navigation system can reliably observe**. Combine anatomical clearance with tracking visibility, calibration uncertainty, acquisition delay and evolving anatomy. Demonstrate when a promising route becomes unsupported, what additional observation could resolve the problem, and how competing tool sequences compare.

Medivis's listed filings establish relevant engineering context around rendering, registration, tracked instruments and navigation feedback. The proposed extensions below are project recommendations, not conclusions about patentable novelty, infringement or freedom to operate. Patent descriptions also do not establish deployed product functionality or clinical performance. Commercial patent decisions require claim and prosecution-history analysis beyond this technical review.

## Scope and publication corrections

The [Medivis list](https://www.medivis.com/patents) contains 28 rows and 27 unique publication identifiers. US11656690B2 appears twice. All unique listed publications were screened through their abstracts and claim text; the registration, navigation, instrument-interaction and visualization filings received closer description and independent-claim review. Hardware and general rendering filings were screened for project relevance. Google Patents reproduces published filing text but qualifies its legal-status and assignee fields; these are not an authoritative current legal-status audit.

Several application publications on the company page have linked grants. Application claims and issued claims can differ, so an A1 identifier should not be treated as the final claim set.

| Listed publication | Related grant checked | Publication date of grant |
|---|---|---|
| US20240412464A1, trace gestures | [US12579761B2](https://patents.google.com/patent/US12579761B2/en) | March 17, 2026 |
| US20240412465A1, automatic registration | [US12456267B2](https://patents.google.com/patent/US12456267B2/en) | October 28, 2025 |
| US20250086899A1, detached visualization | [US12462496B2](https://patents.google.com/patent/US12462496B2/en) | November 4, 2025 |
| US20230320788A1, trajectory navigation | [US12551283B2](https://patents.google.com/patent/US12551283B2/en) | February 17, 2026 |
| US20220354591A1, emanation simulation | [US11744652B2](https://patents.google.com/patent/US11744652B2/en) | September 5, 2023 |

The last pair is the same application at different publication stages. Its dosage concerns radiation from a collimator, not administered anesthetic compounds. US8902224B2 is listed by Medivis, but its public record names Thereitis com Pty Ltd as assignee; the licensing or ownership relationship was not established here.

## What the listed filings describe

These are engineering summaries of selected claim mechanisms, not complete claim constructions. Links lead to the publication text. The opportunity column is a proposed project use, not a claim that the idea is absent from all related patents or literature.

| Publications and topic | Relevant disclosed mechanism | Project implication |
|---|---|---|
| [US11657247B2](https://patents.google.com/patent/US11657247B2/en), adhesive fiducials | Bent reference platform, distinct marker regions, connecting geometry and adhesive backing | Simulate marker visibility, attachment motion and resulting pose error; recreating the physical design is unnecessary |
| [US20250082429A1](https://patents.google.com/patent/US20250082429A1/en), adhesive reference unit | Thin adhesive body with an asymmetric fixed arrangement of marker sites | Test sensitivity to reference placement and reference-to-patient motion |
| [US20240412464A1](https://patents.google.com/patent/US20240412464A1/en), trace alignment | Recorded tool-tip surface trace matched to a virtual surface; the grant adds detailed geometric relationships around traced points | Investigate ambiguity, independent error and which additional measurement is useful |
| [US20240412465A1](https://patents.google.com/patent/US20240412465A1/en), automatic landmarks | Infrared/depth-derived anatomy representation, point-cloud/isosurface matching and reference-array coordinates | Separate successful matching from trustworthy target localization |
| [US20240407856A1](https://patents.google.com/patent/US20240407856A1/en), alignment buffer | Angular/positional departure from planned alignment changes a guide's visual state; buffer boundaries govern transitions | Evaluate an anatomy-dependent clearance decision alongside geometric alignment |
| [US20250086899A1](https://patents.google.com/patent/US20250086899A1/en), detached visualization | Secondary display coordinates for anatomy; the grant includes transforming tracked instrument coordinates into that space | Keep inspection transforms separate from physical coordinates and branch state |
| [US20240412470A1](https://patents.google.com/patent/US20240412470A1/en), distance calibration | Two tracked instruments establish a tip-to-calibration-location distance | Add repeatability measurements, uncertainty propagation and stale-calibration behavior |
| [US11138806B1](https://patents.google.com/patent/US11138806B1/en), distributed rendering | Workstation/headset image composition with headset-side compensation for pose change during latency | Build a replay benchmark with delayed, dropped and reordered observations |
| [US20220354591A1](https://patents.google.com/patent/US20220354591A1/en) and [US11744652B2](https://patents.google.com/patent/US11744652B2/en), emanation and predicted dosage | Tracked collimator beam visualization; granted claim adds planned radiation-dose masks, alignment comparison and cue | Inspiration for displaying spatial effects; a thermal/tissue model needs separate equations and evidence |
| [US11172996B1](https://patents.google.com/patent/US11172996B1/en) and [US12127800B2](https://patents.google.com/patent/US12127800B2/en), instrument registration | Fiducial-to-tip geometry connects physical landmarks and medical-model landmarks in a common frame | Extend the frame graph with measured uncertainty and target-specific error evaluation |
| [US11429247B1](https://patents.google.com/patent/US11429247B1/en), medical slices | Interactive slice panels associated with a medical volume; dependent claims include instrument-oriented views | Compare candidate tools on synchronized slices without making display clipping alter tissue |
| [US11857274B2](https://patents.google.com/patent/US11857274B2/en), marker-bearing instrument | Tubular instrument with code regions at specified geometric relationships to the body and tip | Use generic instrument/reference transforms with explicit calibration |
| [US8902224B2](https://patents.google.com/patent/US8902224B2/en), object display | Metadata positions objects in 3D; offsets reduce clustering while preserving relative ordering | Low priority for surgery simulation; this is not anatomical volume registration |
| [US10952809B2](https://patents.google.com/patent/US10952809B2/en) and [US10265138B2](https://patents.google.com/patent/US10265138B2/en), surgical 3D images | DICOM ordering/orientation, subvolume organization, textures and volume rendering | Preserve spatial fidelity and benchmark rendering cost; avoid making another volume viewer the main contribution |
| [US11395715B2](https://patents.google.com/patent/US11395715B2/en), rendering continuation | Texture sampling and threshold-based subvolume skipping | Keep graphics acceleration separate from quantitative geometry and collision results |
| [US20240241343A1](https://patents.google.com/patent/US20240241343A1/en), snap-on tracking | Body with aligned/off-axis reflectors, grip and attachment-related geometry | Simulate tool identity, calibration changes and occlusion during exchanges |
| [US20240341820A1](https://patents.google.com/patent/US20240341820A1/en), spinous clamp | Opposing gripping arms, adjustable spacing and coupling to another surgical component | Spine-specific hardware; low priority for the glioma project |
| [US11992934B2](https://patents.google.com/patent/US11992934B2/en), stereo video | Medical-model views with multiple video sources in AR | Evaluate consistency across camera sources, calibration and timestamps |
| [US20240238560A1](https://patents.google.com/patent/US20240238560A1/en), catheter snap tool | Marker-bearing body with a longitudinal catheter-retaining groove | Optional later catheter rehearsal; not central to glioma tissue interaction |
| [US20240341907A1](https://patents.google.com/patent/US20240341907A1/en), marker positioner | Rigid polygonal frame, asymmetric marker layout and coupling | Tracking-quality experiment with generic reference geometry |
| [US20230320788A1](https://patents.google.com/patent/US20230320788A1/en), trajectory guide | Pose-driven navigation display; the grant specifies a circle whose size and target indicator encode distance/alignment | Extend route evaluation to complete tools, uncertainty and changing tissue state |
| [US11307653B1](https://patents.google.com/patent/US11307653B1/en), surgical AR input | Virtual hand, gestures and clipping-plane interaction in a unified frame | Keep display commands and operative actions in distinct event channels |
| [US11656690B2](https://patents.google.com/patent/US11656690B2/en), virtual touchpad | Hand movement projected onto a virtual slate control to modify medical-data display | Useful interaction context, but copying the control is weak portfolio differentiation |
| [US11931114B2](https://patents.google.com/patent/US11931114B2/en), virtual instrument interaction | Fiducial-driven virtual instrument, gesture updates, and a virtual offset tip/extension in claim 1 | Distinguish the display extension, actual tool geometry and physical interaction model |

## Recommended project extensions

### 1 Planning with tracking and anatomical uncertainty

**Question:** Does a candidate remain geometrically feasible when the instrument and anatomy are not known exactly?

Estimate the effect of registration, marker localization, tip calibration, observation age, segmentation and supported anatomical displacement on the complete instrument. Preserve correlations: a shared reference error moves both instruments coherently, while a tool-specific calibration error does not. Do not add unrelated uncertainty values as though they were independent standard deviations.

Use bounded perturbations when only bounds are justified, and calibrated distributions when data support them. Evaluate the swept tip and shaft across sampled or bounded states. Return clearance distributions or bounds, unresolved spatial coverage, and the input versions used. These are geometric results under stated assumptions, not probabilities of neurological injury.

The project contribution is a measured chain from uncertain observations to changed planning decisions. A colored halo by itself is insufficient. Compare nominal geometry, a fixed conservative margin, the existing coherent-perturbation baseline and the proposed observation-conditioned model.

Registration quality must use independent target measurements. A low fiducial-fit residual can coexist with poor target accuracy; this distinction is established in [Fitzpatrick et al.](https://pubmed.ncbi.nlm.nih.gov/9874293/). Rigid-registration theory does not establish validity after brain deformation or tissue removal.

**Demonstration:** a nominally clear candidate becomes unsupported after calibration drift or an anatomical update; a different candidate remains feasible within the evaluated bounds. Show why the ranking changed.

### 2 Choosing the next observation

**Question:** Which feasible measurement would most improve this particular decision?

Compare actions such as observing a visible landmark, changing a camera viewpoint, checking calibration or requesting an available imaging update. Score expected improvement in target localization or candidate ranking against acquisition time and feasibility. Do not reward a better surface-fit score alone.

Begin with greedy information selection or short-horizon search. A learned acquisition policy is justified only if it improves a measured trade-off. Random selection, fixed measurement sequences and a greedy information baseline should receive the same observations and opportunities.

Use simulated or physically collected observations for adaptive acquisition experiments. A clinical dataset with a few fixed acquisitions does not allow unlimited new scans at arbitrary times. Keep independent evaluation landmarks out of both measurement selection and fitting.

**Demonstration:** a sparse or ambiguous observation leaves two plausible alignments; the system selects an additional supported measurement that distinguishes them. Include a case where available measurements cannot resolve the ambiguity.

### 3 Bimanual planning that preserves visibility

**Question:** Can both tools execute the maneuver while maintaining useful tracking and operative visibility?

Model camera field of view, marker orientation, line of sight, tool/hand occlusion and observation delay. Keep optical marker visibility separate from seeing tissue through blood or instruments. A marker being visible does not guarantee accurate tracking; image quality and geometric conditioning still matter.

The planner can compare poses, tool exchange order, camera movements and information-gathering pauses. Costs can include time spent with unsupported pose estimates and uncertainty near relevant anatomy, alongside physical clearance and task progress.

First use deterministic visibility geometry and measured tracking errors. Machine learning can later estimate tracking quality from images or rank sequences. Evaluate these outputs against known poses in synthetic scenes and a generic calibration phantom before transferring them to patient-shaped scenes.

**Demonstration:** one sequence has favorable nominal clearance but repeatedly loses tracking when the second hand/tool enters. Another retains useful observations. An oracle baseline may know true pose, but the practical policy must not receive it.

### 4 Cumulative instrument effects and branch comparison

**Question:** How does the history of tool use change the consequences of the next maneuver?

Add persistent cavity, fluid, visibility, thermal-exposure and vessel-patency states. A selected instrument action changes those states, and subsequent actions inherit the changes. Keep tissue removal, thermal exposure, visible blood and regional circulation as distinct quantities. A time slider should reveal the evolving fields and their provenance.

Snapshot the complete simulator before a branch: geometry, both tools, physiology, drug state, pending events and random-state identity. Compare alternatives with the same initial evidence and exogenous scenarios. Retain separate endpoints rather than turning all outcomes into a single compensatory score.

Start with explicitly authored response laws. Physical claims require calibration: aspiration pressure/flow and removed mass; temperature measurements for energy deposition; independent vessel flow for perfusion effects. Do not synthesize postoperative DWI or functional outcomes and present them as clinically predicted consequences.

Systemic physiology and pharmacology can later add context to the branch comparison. The radiation patent does not supply a drug-response model or establish the novelty of adding one. Refer to the [operative dossier](glioma-operative-research-dossier.md) for the necessary distinctions between blood loss, systemic monitors and local perfusion.

### 5 Reproducible navigation failure testing

**Question:** Does the entire application behave correctly when observations fail or arrive late?

Replay a case while injecting controlled marker loss, reference displacement, calibration replacement, timestamp skew, dropped/reordered frames and an unsupported anatomical update. Measure detection delay, false alarms, stale-result exposure, recovery time and re-evaluation cost. Record both simulation time and wall-clock processing time.

The anatomy, live instrument state, speculative branches and each display view must retain distinct identities. Scaling or detaching a display cannot alter physical geometry. Late responses cannot overwrite a newer case or transform. A recovered camera stream does not automatically make an old anatomical registration valid again.

This is achievable without reproducing a headset or proprietary tracking accessory. A local pose/image replay adapter and explicit failure controls provide a relevant systems demonstration. An actual Medivis integration remains contingent on an agreed interface; no public SDK contract was verified in this review.

## Fit with the current repository

| Existing capability | Preserve it | Proposed addition |
|---|---|---|
| [Complete-tool geometry](../src/resectionlab/geometry.py) | Physical frames, shaft/tip checks and conservative motion envelopes | Observation-conditioned tool pose and geometry uncertainty |
| [Coherent uncertainty worlds](../src/resectionlab/worlds.py) | Frozen assumptions and separate optimization/evaluation/stress roles | Time-varying observations, correlated frame errors and explicit calibration provenance |
| [Functional exposure evaluation](../src/resectionlab/functional_events.py) | Evidence coverage, source identity and independent sequence evaluation | Additional separately calibrated effect channels; no generic brain-health score |
| [Sequential simulator](../src/resectionlab/simulation.py) | Existing action semantics and historical benchmark reproducibility | Separate dynamic environment with two persistent tools, duration and interruptions |
| Existing MPR, 3D comparison and replay | Source identity and stale-response protection | Linked evidence, uncertainty and consequence timeline |

The repository already contains uncertainty perturbations and complete-tool checking. Calling those ideas new would obscure the real work. The next increment is connecting uncertainty to incoming observations, instrument visibility and decisions over time.

## First implementation and portfolio demonstration

Implement one local experiment combining recommendations 1, 3 and 5:

1. Load one eligible public research case with declared anatomical evidence and two generic instruments.
2. Replay a scripted, timestamped tracking stream and independently known perturbations.
3. Compare nominal, fixed-margin and observation-conditioned candidate evaluation.
4. Introduce occlusion or calibration drift and expose the resulting loss of support immediately.
5. Re-evaluate after a valid new observation; also demonstrate a case that remains unresolved.
6. Export the input identities, assumptions, measured error, timing, candidate rankings and failure record.

For the ML contribution, choose either tracking-quality estimation or next-observation selection after the baseline identifies a bottleneck. Hold out cases, trajectories and failure combinations as appropriate; do not claim unseen-patient evidence from many correlated frames of one case. Optimize and evaluate on different scenario draws, and retain a separate model-mismatch stress test.

A concise future portfolio description is: **“A glioma planning simulator that evaluates complete instrument sequences under changing anatomical and tracking uncertainty, explains when a plan loses support, and tests which new observation would help.”** Use measured results and a reproducible failure case to substantiate that description once implemented.

This extends the [Medivis project blueprint](medivis-project-blueprint.md). It prioritizes a bounded, testable contribution before whole-operation tissue physics or joint surgical/anesthesia optimization.
