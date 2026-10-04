# HBE numerical evaluation interface

These helpers implement the [fixed specimen protocol](hbe-specimen-mechanics-poc.md). They do not launch a solver. Runtime, model, data access and actual execution still require their separate frozen release. Analytical unit fixtures are not measured specimen results.

## Primitive readout

`scripts/mechanics_hbe_readout.py:read_run` takes a repository root, six `{path, sha256}` bindings (`mesh`, `deck`, `loading`, `nodes`, `elements`, `solver`), the protocol hash and expected branch/N/steps/modulus. It returns `(receipt, cache)`; the optional cache holds current coordinates, raw reactions and an immutable mesh for scale comparisons. Keep these outputs in ignored storage.

The helper requires every actual initial/converged primitive state, verifies declared loading and source identities, reconstructs signed force/torque, deformation Jacobians, three-field energy and 75 probe displacements, and checks final solver residuals. Nonpositive logged Jacobians also reject the run even if node-based samples are positive. Logged `sed` is only diagnostic: it averages pointwise material energy and is not generally the three-field total energy.

`mechanics_hbe_outputs.py` owns the streaming primitive/solver parsers. Primitive times use the recorder's 12 significant digits; nonlinear solver status uses `%lg` precision. No missing state is manufactured. `mechanics_hbe_physics.py` owns independent hex8 kinematics, quadrature and numerical comparisons; it contains no equilibrium solver or mesh generator.

## Comparison inventory

`build_numerical_evidence(runs, caches, source_bindings=..., protocol_sha256=..., fitted_mu_Pa=None)` builds the 18 pre-fit comparisons. Supplying fitted μ requires all 20 runs and adds calibrated axial proportionality confirmation.

Run keys are `branch:N{N}:S{steps}:{role}`. Roles are `reference`, `double_mu`, `fitted`; `mechanics_hbe_access.expected_runs()` supplies the exact inventory. Retain fine-120 caches for reference compression/tension/positive torsion, the two doubled-modulus runs and the two fitted axial runs. Other run receipts need only reaction/probe series and compact numeric criteria.

The evidence groups are `runs`, `mesh` (four branches), `step` (four), `scale` (compression/positive torsion) and, when complete, `fitted_confirmation` (compression/tension). Each criterion records actual error, declared limit and units; strict refinement trends carry an explicit comparison. An aggregate success boolean cannot replace these observations. Sources are bound as `physics`, `primitive_parser`, `mesh_deck`, `curve_evaluator` and `run_readout`.

## Calibration and held-out boundary

`ReleasedStudy(root, protocol_binding, release_binding, ledger_path=...)` reads declarations only. The root-issued calibration release names the exact two members and binds source commit/archive, protocol, roles, acquisition archive and runtime/analytical/mesh prerequisites. Orchestration-specific release fields may be added.

The release fixes the access ledger location and CSV interpretations. Its source archive must contain the bytes of the actually imported helpers and identify the committed revision. Historical mesh preparation may name its original builder separately through `execution.mesh_preparation_source`; the current fitted-deck builder remains independently bound. Each run's execution receipt binds its runtime, command, working directory, deck and primitive outputs.

`read_calibration(schemas)` is the only calibration-member read. Each branch requires explicit delimiter, exact header (or declared headerless data), coordinate/response column indices and units. The reader guesses no header, unit conversion, sign or offset. A schema discrepancy stops the task; it does not select another specimen.

`mechanics_hbe_evaluation.fit_scale` receives the two calibration curves and independently read fine-120 reference axial curves. Save its exact JSON result. Save torsion predictions in `hbe-torsion-prediction-v1` with `reference` and `fitted` dictionaries: each branch has `coordinate` and `response` arrays; fitted rows additionally declare `response_unit: "Nm"`. Reference curves come from the numerical evidence, and fitted predictions are exactly reference response times the frozen scale.

`check_numeric_report` checks freshly calculated comparisons promptly during orchestration. This inexpensive check alone cannot authorize validation access. `freeze_predictions(fit_binding=..., prediction_binding=..., numerical_binding=..., output_path=...)` independently replays all 20 runs from their primitive outputs, reconstructs the comparisons, recomputes the calibration fit and torsion predictions, then exclusively saves the parameter/prediction freeze and verified file inventory.

`evaluate_held_out(freeze_binding=..., schemas=...)` requires the exact saved freeze in the release-bound ledger and verifies all frozen files before reading only the two torsion members. It does not repeat the expensive numerical replay. The durable ledger records and flushes access attempts before member reads as well as completions; a partial reveal counts as exposure. Restarting with a fresh ledger, refitting, or creating another freeze cannot reopen this study after exposure.

The access/readout integration passed 34 independent and owner controls; the [source-bound review receipt](../artifacts/mechanics/hbe-access-independent-review-v1/integrated-final-check.json) records the exact files. Earlier failures and their corrections are preserved in the same review directory. These controls use analytical fixtures and establish no measured specimen fidelity. Hash-bound summaries alone cannot replace primitive reconstruction. This is a reproducibility boundary on a cooperative local host, not an adversarial operating-system security mechanism.
