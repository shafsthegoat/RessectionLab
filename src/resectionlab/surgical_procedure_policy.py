"""Source-bound permission and harm contract for one supratentorial task.

Declarations precede planning. A proposed corridor is never evidence of free
space. Exact source memberships survive categorical ownership and removal.
The thresholds are engineering budgets, not clinical injury thresholds.
"""
from __future__ import annotations

import copy
import math
import re
import numpy as np

from .core import semantic_digest, array_digest
from .surgical_rl_protection import (
    ProtectionAnnotation, ProtectionSet, declared_scene_protection, grid_binding,
)

PROPERTY = 'surgicalProcedurePolicyV1'
SCHEMA = 'resectionlab.supratentorial-procedure-policy.v1'
REQUIRED_STRUCTURES = frozenset(('brainstem', 'cerebellum'))
MAX_CELLS = 2_000_000
_SHA = re.compile('[a-f0-9]{64}')


def procedure_source_binding(scene):
    """Bind anatomy and source provenance before policy-bearing case hashing.

    caseHash includes this policy when persisted in CaseData metadata. The case
    adapter separately binds its complete pre-policy manifest, including pixels
    and source references, so no anatomy or source identity is omitted here.
    """
    source = {key: value for key, value in scene.source.items() if key != 'procedurePolicyBaseCaseHash'}
    if 'procedurePolicyBaseCaseHash' in scene.source:
        source['caseHash'] = scene.source['procedurePolicyBaseCaseHash']
    return semantic_digest({'schema': 'resectionlab.procedure-source-binding.v1',
        'labels': array_digest(scene.labels), 'affine': scene.affine.tolist(),
        'regions': scene.regions, 'source': source})


def _cells(value, scene, name, *, nonempty=False):
    if type(value) is dict and set(value) == {'runs'}:
        runs = value['runs']
        if (type(runs) is not list or any(type(run) is not list or len(run) != 2 or
                any(type(v) is not int for v in run) or not 0 <= run[0] < run[1] <= scene.labels.size
                for run in runs) or any(a[1] >= b[0] for a, b in zip(runs, runs[1:]))):
            raise ValueError('PROCEDURE_INVALID_' + name.upper() + ': ordered disjoint source runs required')
        if sum(stop-start for start, stop in runs) > MAX_CELLS:
            raise ValueError('PROCEDURE_SOURCE_CELL_BOUND_EXCEEDED')
        value = [c for start, stop in runs for c in range(start, stop)]
    if (type(value) is not list or len(value) > MAX_CELLS or
            any(type(c) is not int or not 0 <= c < scene.labels.size for c in value) or
            value != sorted(set(value)) or (nonempty and not value)):
        raise ValueError('PROCEDURE_INVALID_' + name.upper() + ': sorted exact source-cell indices required')
    return frozenset(value)


def _packed(cells):
    """Lossless source-index runs keep full observed-grid declarations compact."""
    values = list(cells)
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError('PROCEDURE_EXACT_SOURCE_CELL_IDS_REQUIRED')
    values = sorted(set(map(int, values)))
    runs = []
    for value in values:
        if runs and runs[-1][1] == value:
            runs[-1][1] = value + 1
        else:
            runs.append([value, value + 1])
    return {'runs': runs}


def _evidence(row, name):
    if (type(row) is not dict or type(row.get('provenance')) is not str or not row['provenance'].strip() or
            type(row.get('evidenceSha256')) is not str or not _SHA.fullmatch(row['evidenceSha256']) or
            row.get('origin') != 'independent-source-declaration'):
        raise ValueError('PROCEDURE_INDEPENDENT_EVIDENCE_REQUIRED: ' + name)


def _finite_nonnegative(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def make_procedure_policy(scene, *, policy_id, evidence, structures,
        intracranial_cells, observed_cells, entry_regions, permitted_compartments,
        work_domain, tissue_permissions, neural_max_mm3, structure_max_mm3,
        start_stage='post-access', bone_max_mm3=None,
        preparation_nonbone_max_mm3=None, modeled_time_max_seconds=None,
        neural_cut_max_mm3=0., structure_cut_max_mm3=None):
    """Serialize independently supplied source declarations; infer no anatomy.

    Every evidence/region row supplies provenance, evidenceSha256 and
    origin='independent-source-declaration'. Structure rows also supply kind
    ('neural' or 'nonneural'), cells, and coverageCells. Coverage is the domain
    in which a negative structure annotation is meaningful, not its foreground.
    """
    record = {'schema': SCHEMA, 'id': policy_id, 'workflow': 'supratentorial-tumor-resection',
        'sourceBinding': procedure_source_binding(scene), 'evidence': copy.deepcopy(evidence),
        'startStage': start_stage, 'structures': copy.deepcopy(structures),
        'intracranialCells': _packed(intracranial_cells),
        'observedCells': _packed(observed_cells),
        'entryRegions': copy.deepcopy(entry_regions),
        'permittedCompartments': copy.deepcopy(permitted_compartments),
        'workDomain': copy.deepcopy(work_domain), 'tissuePermissions': copy.deepcopy(tissue_permissions),
        'maximumNeuralRemovalMm3': neural_max_mm3,
        'maximumBoneRemovalMm3': bone_max_mm3,
        'maximumNonbonePreparationRemovalMm3': preparation_nonbone_max_mm3,
        'maximumModeledElapsedSeconds': modeled_time_max_seconds,
        'maximumNeuralCutSourceMm3': neural_cut_max_mm3,
        'maximumStructureCutSourceMm3': copy.deepcopy(structure_cut_max_mm3 if structure_cut_max_mm3 is not None else
            {'brainstem': 0., 'cerebellum': 0.}),
        'maximumStructureRemovalMm3': copy.deepcopy(structure_max_mm3),
        'selectionRule': 'max_treatment_within_harm_bounds',
        'scenarios': ['nominal'], 'clinicalValidation': False}
    for row in record['structures'].values():
        row['cells'] = _packed(row['cells'])
        row['coverageCells'] = _packed(row['coverageCells'])
    for key in ('entryRegions', 'permittedCompartments'):
        for row in record[key].values():
            row['cells'] = _packed(row['cells'])
    record['policyHash'] = semantic_digest(record)
    return record


def _validated(scene):
    """Return a checked immutable-source declaration or raise a specific code."""
    scene.require_cache_source()
    # Admission and policy validation must classify the same immutable source.
    # Do this before the parsed-policy cache, including its CaseData proof.
    from .surgical_planning_admission import source_classification
    if PROPERTY in scene.properties and source_classification(scene) != 'generated-engineering':
        raise ValueError('PROCEDURE_ACQUIRED_SOURCE_QUALIFICATION_UNAVAILABLE')
    cached = getattr(scene, '_procedure_policy_parsed', None)
    if cached is not None:
        return cached
    raw = scene.properties.get(PROPERTY)
    if raw is None:
        return None
    if type(raw) is not dict or raw.get('schema') != SCHEMA:
        raise ValueError('PROCEDURE_VERSION_UNSUPPORTED')
    if scene.labels.size > MAX_CELLS:
        raise ValueError('PROCEDURE_SOURCE_CELL_BOUND_EXCEEDED')
    if raw.get('sourceBinding') != procedure_source_binding(scene):
        raise ValueError('PROCEDURE_SOURCE_MISMATCH')
    if raw.get('policyHash') != semantic_digest({k: v for k, v in raw.items() if k != 'policyHash'}):
        raise ValueError('PROCEDURE_POLICY_HASH_MISMATCH')
    if (type(raw.get('id')) is not str or not 0 < len(raw['id']) <= 100 or
            raw.get('workflow') != 'supratentorial-tumor-resection' or
            raw.get('startStage') not in ('post-scalp', 'post-access') or
            raw.get('clinicalValidation') is not False or
            raw.get('selectionRule') != 'max_treatment_within_harm_bounds' or raw.get('scenarios') != ['nominal']):
        raise ValueError('PROCEDURE_WORKFLOW_UNSUPPORTED')
    _evidence(raw.get('evidence'), 'procedure assumptions and intracranial/observed coverage')
    if raw['evidence'].get('reviewStatus') not in ('reviewed', 'generated-engineering'):
        raise ValueError('PROCEDURE_SOURCE_REVIEW_REQUIRED')
    review_status = raw['evidence']['reviewStatus']
    intracranial = _cells(raw.get('intracranialCells'), scene, 'intracranial_coverage', nonempty=True)
    observed = _cells(raw.get('observedCells'), scene, 'observed_coverage', nonempty=True)
    if not intracranial <= observed:
        raise ValueError('PROCEDURE_INTRACRANIAL_COVERAGE_INCOMPLETE')
    target_labels = {r['id'] for r in scene.regions if r.get('target')}
    targets = set(map(int, np.flatnonzero(np.isin(scene.labels, list(target_labels)))))
    from .surgical_scene import original_target_memberships
    originals = original_target_memberships(scene)
    if originals is not None:
        targets.update(int(cell) for cells in originals.values() for cell in cells)
    if not targets <= intracranial:
        raise ValueError('PROCEDURE_TARGET_OUTSIDE_INTRACRANIAL_COVERAGE')
    structures = raw.get('structures')
    if type(structures) is not dict or len(structures) > 64:
        raise ValueError('PROCEDURE_STRUCTURE_EVIDENCE_REQUIRED')
    for required in sorted(REQUIRED_STRUCTURES):
        if required not in structures:
            raise ValueError('PROCEDURE_MISSING_' + required.upper() + '_COVERAGE')
    neural = set(); structure_cells = {}
    for name, row in structures.items():
        if type(name) is not str or not 0 < len(name) <= 100 or ':' in name:
            raise ValueError('PROCEDURE_STRUCTURE_ID_INVALID')
        _evidence(row, name)
        if row.get('reviewStatus') != review_status:
            raise ValueError('PROCEDURE_STRUCTURE_REVIEW_REQUIRED: ' + name)
        cells = _cells(row.get('cells'), scene, name, nonempty=name in REQUIRED_STRUCTURES)
        structure_cells[name] = cells
        coverage = _cells(row.get('coverageCells'), scene, name + '_coverage', nonempty=True)
        if (not cells <= coverage or not intracranial <= coverage or
                not coverage <= observed or not cells <= intracranial):
            raise ValueError('PROCEDURE_INCOMPLETE_STRUCTURE_COVERAGE: ' + name)
        if row.get('kind') not in ('neural', 'nonneural'):
            raise ValueError('PROCEDURE_STRUCTURE_TISSUE_KIND_REQUIRED: ' + name)
        if name in REQUIRED_STRUCTURES and row['kind'] != 'neural':
            raise ValueError('PROCEDURE_CRITICAL_NEURAL_KIND_REQUIRED: ' + name)
        if row['kind'] == 'neural':
            neural.update(cells)
    entries, permitted = set(), set()
    for key, destination in (('entryRegions', entries), ('permittedCompartments', permitted)):
        rows = raw.get(key)
        if type(rows) is not dict or not rows:
            raise ValueError('PROCEDURE_' + key.upper() + '_REQUIRED')
        for name, row in rows.items():
            _evidence(row, key + ':' + name)
            cells = _cells(row.get('cells'), scene, key, nonempty=True)
            if not cells <= observed:
                raise ValueError('PROCEDURE_DECLARATION_OUTSIDE_OBSERVED_COVERAGE: ' + key)
            destination.update(cells)
    if not entries <= permitted or not entries & intracranial:
        raise ValueError('PROCEDURE_ENTRY_OUTSIDE_PERMITTED_COMPARTMENTS')
    domain = raw.get('workDomain')
    _evidence(domain, 'work domain')
    from .surgical_task_domain import TaskDomain
    bounds = []
    for key in ('minimumVoxelBoundary', 'maximumVoxelBoundary'):
        value = domain.get(key)
        if (type(value) is not list or len(value) != 3 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
            raise ValueError('PROCEDURE_WORK_DOMAIN_INVALID')
        bounds.append(np.asarray(value, float))
    low, high = bounds
    if np.any(low < -.5) or np.any(high > np.asarray(scene.shape)-.5) or np.any(low >= high):
        raise ValueError('PROCEDURE_WORK_DOMAIN_OUTSIDE_OBSERVED_GRID')
    permissions = raw.get('tissuePermissions')
    materials = {r['material'] for r in scene.regions}
    if (type(permissions) is not dict or not materials <= permissions.keys() or
            any(type(v) is not dict or set(v) != {'remove', 'cut'} or
                any(type(flag) is not bool for flag in v.values()) for v in permissions.values())):
        raise ValueError('PROCEDURE_EXPLICIT_TISSUE_PERMISSIONS_REQUIRED')
    # Soft tissue is mechanically broad. Its permitted removal must have an
    # independently classified source membership, never a name-derived class.
    classified = set().union(*structure_cells.values())
    soft = {r['id'] for r in scene.regions if r['material'] == 'soft_tissue'}
    permitted_soft = {c for c in permitted if int(scene.labels.flat[c]) in soft}
    if not permitted_soft <= classified:
        raise ValueError('PROCEDURE_UNCLASSIFIED_SOFT_TISSUE')
    limits = raw.get('maximumStructureRemovalMm3')
    if (not _finite_nonnegative(raw.get('maximumNeuralRemovalMm3')) or type(limits) is not dict or
            not REQUIRED_STRUCTURES <= limits.keys() or any(k not in structures or not _finite_nonnegative(v) for k, v in limits.items()) or
            any(limits[k] != 0 for k in REQUIRED_STRUCTURES)):
        raise ValueError('PROCEDURE_EXPLICIT_NEURAL_BUDGET_REQUIRED')
    if any(not _finite_nonnegative(raw.get(name)) for name in (
            'maximumBoneRemovalMm3', 'maximumNonbonePreparationRemovalMm3', 'maximumModeledElapsedSeconds',
            'maximumNeuralCutSourceMm3')):
        raise ValueError('PROCEDURE_EXPLICIT_COMPLETE_SEQUENCE_BUDGET_REQUIRED')
    cut_limits = raw.get('maximumStructureCutSourceMm3')
    if (type(cut_limits) is not dict or not REQUIRED_STRUCTURES <= cut_limits.keys() or
            any(name not in structures or not _finite_nonnegative(value) for name, value in cut_limits.items()) or
            any(cut_limits[name] != 0 for name in REQUIRED_STRUCTURES)):
        raise ValueError('PROCEDURE_EXPLICIT_STRUCTURE_CUT_BUDGET_REQUIRED')
    parsed = {'record': raw, 'intracranial': intracranial, 'observed': observed,
        'entries': frozenset(entries), 'permitted': frozenset(permitted), 'neural': frozenset(neural),
        'structureCells': structure_cells,
        'domain': TaskDomain(tuple(low), tuple(high), domain['evidenceSha256'], domain['evidenceSha256'])}
    scene._procedure_policy_parsed = parsed
    return parsed


def procedure_admission(scene, protection=None):
    present = PROPERTY in scene.properties
    try:
        policy = _validated(scene)
        if policy is None:
            return {'present': False, 'available': False, 'policyId': None, 'policyHash': None,
                'blockers': [{'code': 'PROCEDURE_POLICY_REQUIRED', 'message': 'The supported supratentorial task requires an independent source-bound procedure policy.'}]}
        if protection is not None:
            if protection.scene.identity != scene.identity:
                raise ValueError('PROCEDURE_PROTECTION_SOURCE_MISMATCH')
            missing = set(protection.missing) - REQUIRED_STRUCTURES
            if missing:
                raise ValueError('PROCEDURE_EXISTING_REQUIRED_COVERAGE_MISSING: ' + ', '.join(sorted(missing)))
        raw = policy['record']
        return {'present': True, 'available': True, 'policyId': raw['id'], 'policyHash': raw['policyHash'],
            'schema': SCHEMA, 'startStage': raw['startStage'], 'blockers': [],
            'requiredCoverage': ['brainstem', 'cerebellum', 'intracranial'],
            'scenarios': ['nominal'], 'clinicalValidation': False,
            'limitations': ['Nominal deterministic source scenario only; no uncertainty distribution or functional injury evaluator.',
                'Post-access or post-scalp starting costs before the supplied source checkpoint are unknown.']}
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        reason = str(error)
        return {'present': present, 'available': False, 'policyId': None, 'policyHash': None,
            'blockers': [{'code': reason.split(':')[0], 'message': reason}]}


def _boundary(scene, cells):
    mask = np.zeros(scene.shape, bool)
    mask.ravel()[list(cells)] = True
    interior = mask.copy()
    for axis in range(3):
        for shift in (-1, 1):
            neighbor = np.roll(mask, shift, axis=axis)
            index = [slice(None)] * 3; index[axis] = 0 if shift == 1 else -1
            neighbor[tuple(index)] = False
            interior &= neighbor
    return set(map(int, np.flatnonzero(mask & ~interior)))


def procedure_protection(scene, protection=None):
    """Enlarge exclusions with exact critical, compartment and unknown cells."""
    policy = _validated(scene)
    protection = declared_scene_protection(scene, protection)
    if policy is None:
        return protection
    raw = policy['record']; binding = grid_binding(scene)
    rows = {} if protection is None else {a.structure_id: a for a in protection.annotations}
    for name in sorted(REQUIRED_STRUCTURES):
        row = raw['structures'][name]
        old = rows.get(name)
        cells = set(policy['structureCells'][name]) | (set(old.cells) if old else set())
        if old is None or not cells <= set(old.cells):
            rows[name] = ProtectionAnnotation(name, tuple(sorted(cells)), binding,
                row['provenance'] + ('; ' + old.provenance if old else ''), row.get('reviewStatus', 'unreviewed'), name)
    # All unobserved source cells are unknown. Intracranial traversal is allowed
    # only in independently declared compartments and through declared entries.
    excluded = policy.get('exclusions')
    if excluded is None:
        excluded = set(range(scene.labels.size)) - policy['observed']
        excluded.update(policy['intracranial'] - policy['permitted'])
        excluded.update(set(map(int, np.flatnonzero(scene.labels))) - policy['permitted'])
        excluded.update(_boundary(scene, policy['intracranial']) - policy['entries'])
        policy['exclusions'] = frozenset(excluded)
    excluded = set(excluded)
    identity = 'procedure-forbidden-or-unknown'
    old = rows.get(identity)
    if old is not None:
        excluded.update(old.cells)
    if old is None or not excluded <= set(old.cells):
        rows[identity] = ProtectionAnnotation(identity, tuple(sorted(excluded)), binding,
            'Independent observed coverage, intracranial compartments and entry regions; unknown is forbidden',
            raw['evidence']['reviewStatus'])
    required = set(REQUIRED_STRUCTURES) | (set(protection.required) if protection is not None else set())
    if (protection is not None and required == set(protection.required) and
            len(rows) == len(protection.annotations) and all(rows[a.structure_id] is a for a in protection.annotations)):
        return protection
    return ProtectionSet(scene, rows.values(), clearance_mm=protection.clearance_mm if protection else 0.,
        required=tuple(sorted(required)))


def _refusal(engine, code, message, *, violations=None):
    return {'status': 'rejected', 'reason': code + ': ' + message, 'reasonCode': code,
        'completedFraction': 0., 'elapsedSeconds': 0., 'stateId': engine.state_id,
        'events': [], 'deltas': {}, 'metrics': engine.metrics(),
        'constraintViolations': violations or [{'code': code, 'message': message}]}


def procedure_action_rejection(engine, action, *, check=lambda: None):
    if PROPERTY not in engine.scene.properties:
        return None
    admission = procedure_admission(engine.scene, engine.protection)
    if not admission['available']:
        block = admission['blockers'][0]
        return _refusal(engine, block['code'], block['message'])
    policy = _validated(engine.scene)
    from .surgical_task_domain import action_rejection, _posed, _outside
    if (type(action) is dict and action.get('slot') in engine.state['slots'] and
            action.get('operation') in ('inspect', 'deactivate', 'release') and not _posed(engine, action['slot'])):
        from .surgical_motion import envelopes
        from .surgical_tools import TOOLS
        segments = [segment for key, slot in engine.state['slots'].items() if _posed(engine, key)
            for segment in envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion'])]
        reason = _outside(engine.scene, policy['domain'], segments)
        return _refusal(engine, 'PROCEDURE_WORK_DOMAIN_VIOLATION', reason) if reason else None
    refusal = action_rejection(engine, action, domain=policy['domain'], check=check)
    if refusal is not None:
        return _refusal(engine, 'PROCEDURE_WORK_DOMAIN_VIOLATION', refusal['reason'])
    return None


def procedure_source_memberships(scene):
    """Observable source memberships, shared by search and evaluation adapters."""
    policy = _validated(scene)
    if policy is None:
        raise ValueError('PROCEDURE_POLICY_REQUIRED')
    return {'observed': policy['observed'], 'intracranial': policy['intracranial'],
        'entry': policy['entries'], 'permitted': policy['permitted'], 'neural': policy['neural'],
        **{'structure:' + name: cells for name, cells in policy['structureCells'].items()}}


def direct_removal_dimensions(engine, baseline):
    """Count unique immutable source IDs; overlaps remain separate dimensions."""
    from .surgical_sequence_outcomes import _quantity, _unknown
    scene = engine.scene
    removed, before = set(engine.state['removed']), set(baseline.state['removed'])
    def cuts(owner):
        return {int(c) for field in ('cutEdges', 'transectedEdges') for edge in owner.state.get(field, ()) for c in edge}
    cut, cut_before = cuts(engine), cuts(baseline)
    def quantity(cells):
        return _quantity(len(removed & cells)*scene.voxel_mm3, len(before & cells)*scene.voxel_mm3, 'mm3')
    def cut_quantity(cells):
        result = _quantity(len(cut & cells)*scene.voxel_mm3, len(cut_before & cells)*scene.voxel_mm3, 'mm3')
        result['meaning'] = 'Unique source-cell membership incident to existing native cut/transection edges; not physical cut volume or neurological injury.'
        return result
    bone_ids = {r['id'] for r in scene.regions if r['material'] == 'bone'}
    target_ids = {r['id'] for r in scene.regions if r.get('target')}
    bone = {c for c in removed | before if int(scene.labels.flat[c]) in bone_ids}
    nonbone = {c for c in removed | before if int(scene.labels.flat[c]) not in bone_ids | target_ids}
    dims = {'boneRemovedMm3': quantity(bone), 'nonbonePreparationRemovedMm3': quantity(nonbone)}
    policy = _validated(scene)
    if policy is None:
        dims['neuralRemovedMm3'] = _unknown('Explicit complete source neural memberships are unavailable.', 'mm3')
        dims['neuralCutSourceMm3'] = _unknown('Explicit complete source neural memberships are unavailable.', 'mm3')
        dims['nonbonePreparationRemovedMm3'] = _unknown('Cannot separate neural tissue from nonbone preparation without complete source neural memberships.', 'mm3')
        dims['structureRemovedMm3'] = {}
        dims['structureCutSourceMm3'] = {}
    else:
        dims['neuralRemovedMm3'] = quantity(set(policy['neural']))
        dims['neuralCutSourceMm3'] = cut_quantity(set(policy['neural']))
        dims['nonbonePreparationRemovedMm3'] = quantity(nonbone - policy['neural'])
        memberships = {name: set(cells) for name, cells in policy['structureCells'].items()}
        if engine.protection is not None:
            for annotation in engine.protection.annotations:
                if annotation.structure_id != 'procedure-forbidden-or-unknown':
                    memberships.setdefault(annotation.structure_id, set()).update(annotation.cells)
        dims['structureRemovedMm3'] = {name: quantity(cells) for name, cells in memberships.items()}
        dims['structureCutSourceMm3'] = {name: cut_quantity(cells) for name, cells in memberships.items()}
    protection = engine.protection
    dims['nonbonePreparationRemovedMm3']['meaning'] = 'Unique non-target, nonbone, nonneural source removal across all accepted stages; no prefix refund.'
    dims['protectedRemovedMm3'] = (_unknown('No source-bound protected memberships supplied.', 'mm3')
        if protection is None else quantity(set(map(int, protection.cells))))
    return dims


def evaluate_procedure_sequence(engine, baseline=None):
    from .surgical_sequence_outcomes import sequence_outcome
    admission = procedure_admission(engine.scene, engine.protection)
    if not admission['available']:
        return {'admissible': False, 'violations': admission['blockers'], 'outcome': None,
            'policyId': admission['policyId'], 'policyHash': admission['policyHash']}
    outcome = sequence_outcome(engine, baseline=baseline)
    policy = _validated(engine.scene)['record']
    violations = _constraint_violations(engine, policy, outcome['dimensions'])
    return {'admissible': not violations, 'violations': violations, 'outcome': outcome,
        'policyId': policy['id'], 'policyHash': policy['policyHash']}


def procedure_transition_assessment(engine, *, source):
    """Check an engine-owned append-only candidate without rebuilding a report.

    Source membership counts use the same complete-sequence dimension builder
    and constraint evaluator as the public outcome. This private transition
    gate does not certify an externally supplied history or publish an outcome.
    """
    admission = procedure_admission(engine.scene, engine.protection)
    if not admission['available']:
        return {'admissible': False, 'violations': admission['blockers']}
    expected = source.state_id
    prefix_valid = (engine.scene is source.scene and engine.protection is source.protection and
        engine.guard_config is source.guard_config and engine._protected_ids is source._protected_ids and
        engine._cranial_stages is source._cranial_stages and
        len(source.actions) == len(source.history) and
        len(engine.actions) == len(engine.history) == len(source.actions) + 1 and
        all(a is b for a, b in zip(engine.actions[:-1], source.actions)) and
        all(a is b for a, b in zip(engine.history[:-1], source.history)) and
        (source.history[-1]['stateAfter'] == expected if source.history else
            source._native_initial_state_id == expected) and
        engine.history[-1]['stateBefore'] == expected and
        engine.history[-1]['stateAfter'] == engine.state_id and
        engine.history[-1]['action'] == engine.actions[-1])
    if not prefix_valid:
        return {'admissible': False, 'violations': [{'code': 'PROCEDURE_TRANSITION_PREFIX_MISMATCH'}]}
    from .surgical_sequence_outcomes import _quantity
    dimensions = direct_removal_dimensions(engine, engine)
    # Only cumulative values enter hard limits. A report's baseline and spatial
    # summaries remain the public complete-sequence evaluator's responsibility.
    elapsed = engine.state['elapsedSeconds']
    dimensions['elapsedSeconds'] = _quantity(elapsed, elapsed, 's', status='modeled_unvalidated')
    policy = _validated(engine.scene)['record']
    violations = _constraint_violations(engine, policy, dimensions)
    return {'admissible': not violations, 'violations': violations,
        'policyId': policy['id'], 'policyHash': policy['policyHash']}


def _constraint_violations(engine, policy, dims):
    """Single hard-constraint definition for public reports and precommit."""
    violations = []
    def maximum(name, dimension, limit):
        value = dimension.get('cumulative')
        if dimension.get('status') == 'unavailable' or value is None:
            violations.append({'code': 'PROCEDURE_REQUIRED_OUTCOME_UNAVAILABLE', 'metric': name})
        elif value > limit + 1e-9:
            violations.append({'code': 'PROCEDURE_HARM_BUDGET_EXCEEDED', 'metric': name,
                'actual': value, 'maximum': limit, 'unit': dimension['unit']})
    maximum('neuralRemovedMm3', dims['neuralRemovedMm3'], policy['maximumNeuralRemovalMm3'])
    maximum('boneRemovedMm3', dims['boneRemovedMm3'], policy['maximumBoneRemovalMm3'])
    maximum('nonbonePreparationRemovedMm3', dims['nonbonePreparationRemovedMm3'], policy['maximumNonbonePreparationRemovalMm3'])
    maximum('elapsedSeconds', dims['elapsedSeconds'], policy['maximumModeledElapsedSeconds'])
    maximum('neuralCutSourceMm3', dims['neuralCutSourceMm3'], policy['maximumNeuralCutSourceMm3'])
    for name, limit in policy['maximumStructureRemovalMm3'].items():
        maximum('structure:' + name + ':removedMm3', dims['structureRemovedMm3'][name], limit)
    for name, limit in policy['maximumStructureCutSourceMm3'].items():
        maximum('structure:' + name + ':cutSourceMm3', dims['structureCutSourceMm3'][name], limit)
    maximum('protectedRemovedMm3', dims['protectedRemovedMm3'], 0.)
    # Tissue permission checks use source ownership for tool compatibility,
    # independently of the overlapping anatomical removal dimensions above.
    removed = set(engine.state['removed'])
    cut = {int(c) for field in ('cutEdges', 'transectedEdges') for edge in engine.state.get(field, ()) for c in edge}
    for effect, cells in (('remove', removed), ('cut', cut)):
        denied = sorted({engine.by_label[int(engine.scene.labels.flat[c])]['material'] for c in cells
            if not policy['tissuePermissions'][engine.by_label[int(engine.scene.labels.flat[c])]['material']][effect]})
        if denied:
            violations.append({'code': 'PROCEDURE_TISSUE_EFFECT_FORBIDDEN', 'effect': effect, 'materials': denied})
    critical = set().union(*(_validated(engine.scene)['structureCells'][name] for name in REQUIRED_STRUCTURES))
    if critical & cut:
        violations.append({'code': 'PROCEDURE_CRITICAL_CUT_FORBIDDEN', 'sourceCells': sorted(critical & cut)})
    return violations


def procedure_comparison_policy(scene):
    """Existing consequence-comparator declaration for the supported task."""
    raw = _validated(scene)['record']
    constraints = [{'metric': 'neuralRemovedMm3', 'scope': 'cumulative', 'operator': 'at_most',
        'value': raw['maximumNeuralRemovalMm3']},
        {'metric': 'boneRemovedMm3', 'scope': 'cumulative', 'operator': 'at_most', 'value': raw['maximumBoneRemovalMm3']},
        {'metric': 'nonbonePreparationRemovedMm3', 'scope': 'cumulative', 'operator': 'at_most', 'value': raw['maximumNonbonePreparationRemovalMm3']},
        {'metric': 'elapsedSeconds', 'scope': 'cumulative', 'operator': 'at_most', 'value': raw['maximumModeledElapsedSeconds']},
        {'metric': 'neuralCutSourceMm3', 'scope': 'cumulative', 'operator': 'at_most', 'value': raw['maximumNeuralCutSourceMm3']},
        {'metric': 'protectedRemovedMm3', 'scope': 'cumulative', 'operator': 'at_most', 'value': 0.}]
    constraints += [{'metric': 'structure:' + name + ':removedMm3', 'scope': 'cumulative',
        'operator': 'at_most', 'value': value} for name, value in raw['maximumStructureRemovalMm3'].items()]
    constraints += [{'metric': 'structure:' + name + ':cutSourceMm3', 'scope': 'cumulative',
        'operator': 'at_most', 'value': value} for name, value in raw['maximumStructureCutSourceMm3'].items()]
    return {'id': raw['id'], 'basis': 'declared_research_task_not_clinical_threshold',
        'objectives': ['targetRemovedMm3', 'neuralRemovedMm3', 'neuralCutSourceMm3', 'boneRemovedMm3',
            'nonbonePreparationRemovedMm3', 'elapsedSeconds'] +
            ['structure:' + name + ':' + effect for name in raw['structures'] for effect in ('removedMm3', 'cutSourceMm3')],
        'constraints': constraints, 'requiredMetrics': ['neuralRemovedMm3', 'boneRemovedMm3', 'protectedRemovedMm3']}
