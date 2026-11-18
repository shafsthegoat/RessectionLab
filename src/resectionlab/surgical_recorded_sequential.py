"""Exact transport adapter for previously recorded generated finite-skill scenes.

The native source remains the original generated scene. A display CaseData is a
separate synthetic transport, never a new patient or a trajectory identity
migration. No metadata can change the generator, anatomy, policy or physics.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import re
from pathlib import Path

import numpy as np

from .core import CaseData, SourceRef, array_digest, semantic_digest, thaw_json

PROPERTY = 'recordedSequentialEpisodeV1'
SCHEMA = 'resectionlab.recorded-sequential-source.v1'
PRESERVATION_SCHEMA = 'resectionlab.recorded-sequential-preservation-source.v1'
_METADATA = {'generated', 'is_synthetic', 'surgical_adapter', 'surgical_properties',
             'anatomy_layers', PROPERTY}
_ADAPTER = {'type': 'generated-engineering-control', 'recordedSequentialSource': True}
_EXPECTED_SOURCE = {'kind':'generated-control',
    'provenance':'Prospective generated post-access finite-skill geometry',
    'evidenceClass':'generated-engineering'}


def task_implementation_sha256():
    return hashlib.sha256(Path(__file__).with_name('surgical_sequential_task.py').read_bytes()).hexdigest()


def generator_definition_sha256(name='generated_task'):
    tree = ast.parse(Path(__file__).with_name('surgical_sequential_task.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def declaration(task):
    from .surgical_procedure_policy import procedure_preservation_preference

    scene = task.scene
    value = dict(schema=SCHEMA, parent=task.audit['parent'], sourceSceneId=scene.identity,
        source=copy.deepcopy(scene.source), labelsHash=array_digest(scene.labels),
        affineHash=array_digest(scene.affine), regionsHash=semantic_digest(scene.regions),
        propertiesHash=semantic_digest(scene.properties),
        generatorDefinitionSha256=generator_definition_sha256(),
        taskImplementationSha256AtExport=task_implementation_sha256())
    preference = procedure_preservation_preference(scene)
    if preference is not None:
        value.update(schema=PRESERVATION_SCHEMA, preservationPreference=preference,
            preservationDefinitionSha256=generator_definition_sha256('generated_preservation_task'))
    return {**value, 'contentHash': semantic_digest(value)}


def display_case(task):
    """Package original synthetic source arrays; do not replay or relabel them."""
    scene = task.scene; parent = task.audit['parent']
    return CaseData(case_id=f'generated-sequential-parent-{parent}',
        mri=scene.labels.astype(np.float32), affine=scene.affine,
        compartments={r['name']: scene.labels == r['id'] for r in scene.regions},
        source_refs=(SourceRef('generated', f'synthetic://sequential-task/{parent}', provenance='simulated'),),
        metadata={'generated': True, 'is_synthetic': True,
            'surgical_adapter': copy.deepcopy(_ADAPTER),
            'surgical_properties': copy.deepcopy(scene.properties),
            'anatomy_layers': {r['name']: {'kind': r['material']} for r in scene.regions},
            PROPERTY: declaration(task)})


def restore_recorded_source(case, ordinary_scene, mapping=None):
    """Check actual case provenance first, then reconstruct the original source.

    This branch is available only for this bounded synthetic transport. The
    ordinary adapter/classifier still sees contradictory acquired markers before
    any generator is invoked. Exact metadata fields prevent hidden provenance
    aliases from being dropped by a synthetic declaration.
    """
    from .surgical_planning_admission import source_classification, _acquired_markers
    if source_classification(ordinary_scene) != 'generated-engineering':
        raise ValueError('RECORDED_SEQUENTIAL_GENERATED_SOURCE_REQUIRED')
    metadata = thaw_json(case.metadata)
    if (set(metadata) != _METADATA or metadata['generated'] is not True or
            metadata['is_synthetic'] is not True or metadata['surgical_adapter'] != _ADAPTER):
        raise ValueError('RECORDED_SEQUENTIAL_EXACT_METADATA_REQUIRED')
    raw = metadata[PROPERTY]
    if (type(raw) is not dict or raw.get('schema') not in (SCHEMA, PRESERVATION_SCHEMA) or
            type(raw.get('parent')) is not int or not 0 <= raw['parent'] < 1_000_000 or
            raw.get('contentHash') != semantic_digest({k:v for k,v in raw.items() if k != 'contentHash'})):
        raise ValueError('RECORDED_SEQUENTIAL_SOURCE_BINDING_CHANGED')
    if type(raw.get('source')) is not dict or _acquired_markers(raw['source']) or raw['source'] != _EXPECTED_SOURCE:
        raise ValueError('RECORDED_SEQUENTIAL_GENERATED_SOURCE_REQUIRED')
    parent = raw['parent']
    reference = SourceRef('generated', f'synthetic://sequential-task/{parent}', provenance='simulated')
    if (case.case_id != f'generated-sequential-parent-{parent}' or case.frame != 'RAS+' or
            tuple(r.to_dict() for r in case.source_refs) != (reference.to_dict(),) or
            case.context is not None or case.revision != 1 or bool(case.unknowns) or
            case.brain_mask is not None or case.structural_evidence or case.critical_evidence or
            case.prior_proposals or case.functional_evidence is not None or
            set(case.source_compartments) != set(case.compartments) or
            any(not np.array_equal(case.source_compartments[k],v) for k,v in case.compartments.items())):
        raise ValueError('RECORDED_SEQUENTIAL_EXACT_SYNTHETIC_TRANSPORT_REQUIRED')
    if (raw.get('generatorDefinitionSha256') != generator_definition_sha256() or
            re.fullmatch(r'[0-9a-f]{64}', str(raw.get('taskImplementationSha256AtExport'))) is None):
        raise ValueError('RECORDED_SEQUENTIAL_GENERATOR_DEFINITION_CHANGED')
    from .surgical_sequential_task import generated_preservation_task, generated_task
    if raw['schema'] == PRESERVATION_SCHEMA:
        from .surgical_treatment_intent import validate_preservation_preference

        preference = raw.get('preservationPreference')
        validate_preservation_preference(preference)
        task = generated_preservation_task(parent, preservation_metrics=preference['metrics'])
    else:
        task = generated_task(parent)
    scene = task.scene
    expected = declaration(task)
    # Full file provenance remains historical when unrelated endpoint helpers
    # evolve. Exact generator AST, resulting scene and native replay stay bound.
    expected['taskImplementationSha256AtExport'] = raw['taskImplementationSha256AtExport']
    expected['contentHash'] = semantic_digest({k:v for k,v in expected.items() if k != 'contentHash'})
    if (raw != expected or semantic_digest(metadata['surgical_properties']) != semantic_digest(scene.properties) or
            metadata['anatomy_layers'] != {r['name']:{'kind':r['material']} for r in scene.regions} or
            ordinary_scene.regions != scene.regions or mapping is not None and mapping != scene.regions or
            not np.array_equal(ordinary_scene.labels, scene.labels) or
            not np.array_equal(ordinary_scene.affine, scene.affine) or
            not np.array_equal(case.mri, scene.labels.astype(np.float32)) or
            set(case.compartments) != {r['name'] for r in scene.regions} or
            any(not np.array_equal(case.compartments[r['name']], scene.labels == r['id']) for r in scene.regions)):
        raise ValueError('RECORDED_SEQUENTIAL_SOURCE_CONTENT_CHANGED')
    return scene


def source_geometry(scene):
    """Small original membership overlay, independent of treatment and method."""
    from .surgical_procedure_policy import procedure_source_memberships
    memberships = procedure_source_memberships(scene)
    structures = []
    for name in ('motor-cortex', 'corticospinal', 'brainstem', 'cerebellum'):
        cells = sorted(int(c) for c in memberships['structure:'+name])
        structures.append(dict(id=name, label='Generated '+name, kind='neural',
            evidenceKind='generated-engineering', reviewStatus='generated-engineering',
            provenance='Independent original generated source membership; no functional mapping',
            sourceCells=cells, pointsRasMm=scene.points(cells).tolist() if cells else [],
            displayTruncated=False, removedMm3=0., cutSourceMm3=0.))
    target = sorted(set(int(i) for r in scene.regions if r.get('target')
        for i in np.flatnonzero(scene.labels == r['id'])))
    return dict(structures=structures, targetSourceCells=target,
        targetPointsRasMm=scene.points(target).tolist())


def validate_episode_record(scene, record, case_hash, *, check=None):
    """Cold-admit every real request once; immutable hot playback reuses admission.

    This checks refused requests as well as accepted native branches. Resealing
    changed display totals, tool settings, memberships or source/state bindings
    cannot authenticate them. It grants no new policy or patient permission.
    """
    from .surgical_engine import SurgicalEngine
    from .surgical_sequential_task import COMMON_CONFIG, endpoint_assessment
    from .surgical_procedure_policy import evaluate_procedure_sequence
    p = record.get('sequentialEpisodes')
    if (type(p) is not dict or p.get('schema') != 'resectionlab.recorded-sequential-episodes.v1' or
            p.get('caseHash') != case_hash or p.get('sourceSceneId') != scene.identity or
            p.get('configHash') != COMMON_CONFIG.fingerprint or
            p.get('contentHash') != semantic_digest({k:v for k,v in p.items() if k != 'contentHash'}) or
            p.get('geometry') != source_geometry(scene) or type(p.get('episodes')) is not list or len(p['episodes']) != 5):
        raise ValueError('RECORDED_SEQUENTIAL_PACKET_BINDING_CHANGED')
    signatures = {(r.get('method'), r.get('seed')) for r in p['episodes'] if type(r) is dict}
    if signatures != {('learned',17),('learned',43),('greedy',0),('search',0),('fixed-narrow-greedy',0)}:
        raise ValueError('RECORDED_SEQUENTIAL_METHOD_IDENTITIES_CHANGED')
    branches=set()
    for row in p['episodes']:
        branch = row.get('branchId'); snapshot = record.get('branches', {}).get(branch)
        engine = SurgicalEngine(scene)
        if (branch in branches or snapshot is None or p['sourceStateId'] != engine.state_id or
                p['policyHash'] != engine.snapshot()['procedurePolicy']['policyHash'] or
                semantic_digest(snapshot) != row.get('snapshotHash') or
                type(row.get('attempts')) is not list or len(row['attempts']) > 512):
            raise ValueError('RECORDED_SEQUENTIAL_BRANCH_BINDING_CHANGED')
        branches.add(branch); states=[engine.state_id]; native_position=0; invalid=0
        for index, attempt in enumerate(row['attempts']):
            if check: check()
            action=attempt['action']; slot=engine.state['slots'][action['slot']]
            before=engine.state_id; parameters=copy.deepcopy(engine.parameters(slot))
            instrument=slot['instrument']; position=copy.deepcopy(slot['position'])
            actual=engine.step(copy.deepcopy(action), check=check)
            accepted=actual['status'] in ('accepted','partial'); invalid += int(actual['status']=='rejected')
            expected=dict(attemptIndex=index,nativePositionBefore=native_position,
                nativePositionAfter=native_position+int(accepted),stateBefore=before,stateAfter=engine.state_id,
                effectiveParametersBefore=parameters,effectiveParametersAfter=engine.parameters(engine.state['slots'][action['slot']]),
                instrumentBefore=instrument,instrumentAfter=engine.state['slots'][action['slot']]['instrument'],
                toolPositionBefore=position,toolPositionAfter=engine.state['slots'][action['slot']]['position'],
                status=actual['status'],reason=actual.get('reason'),reasonCode=actual.get('reasonCode'),
                elapsedSeconds=actual['elapsedSeconds'],completedFraction=actual['completedFraction'])
            if any(attempt.get(k) != v for k,v in expected.items()):
                raise ValueError('RECORDED_SEQUENTIAL_REQUEST_CHANGED')
            if accepted: native_position+=1;states.append(engine.state_id)
        stopped = row.get('checkedStop') is True
        endpoint=endpoint_assessment(engine,COMMON_CONFIG,attempts=len(row['attempts'])+int(stopped),
            invalid_actions=invalid,stopped=stopped,externally_truncated=not stopped)
        dimensions=evaluate_procedure_sequence(engine)['outcome']['dimensions']
        cleanup=dict(debrisMm3=endpoint['debrisMm3'],fluidMm3=endpoint['fluidMm3'],
            debrisCollectedMm3=engine.metrics()['debrisRemovedMm3'])
        if (engine.snapshot()!=snapshot or row.get('finalStateId')!=engine.state_id or
                row.get('nativeActions')!=native_position or row.get('acceptedStateIds')!=states or
                row.get('actionHistoryHash')!=semantic_digest(engine.history) or
                row.get('endpoint')!=endpoint or row.get('progress')!=endpoint['progress'] or
                row.get('cleanup')!=cleanup or row.get('modeledSeconds')!=engine.state['elapsedSeconds'] or
                row.get('neuralRemovedMm3')!=dimensions['neuralRemovedMm3']['cumulative'] or
                row.get('structureRemovedMm3')!={k:v['cumulative'] for k,v in dimensions['structureRemovedMm3'].items()} or
                row.get('structureCutSourceMm3')!={k:v['cumulative'] for k,v in dimensions['structureCutSourceMm3'].items()}):
            raise ValueError('RECORDED_SEQUENTIAL_REMEASURED_ENDPOINT_CHANGED')
    return True


def validate_episode_boundary(engine, record):
    """Remeasure the selected accepted endpoint without replaying other branches."""
    from .surgical_sequential_task import COMMON_CONFIG, endpoint_assessment
    packet=record.get('sequentialEpisodes') or {}
    row=next((r for r in packet.get('episodes',[]) if r.get('branchId')==record.get('branchId')),None)
    if row is None: return True
    position=len(engine.actions)
    if position>row['nativeActions'] or engine.state_id!=row['acceptedStateIds'][position]:
        raise ValueError('RECORDED_SEQUENTIAL_SELECTED_BOUNDARY_CHANGED')
    if position==row['nativeActions']:
        endpoint=endpoint_assessment(engine,COMMON_CONFIG,attempts=len(row['attempts'])+int(row['checkedStop']),
            invalid_actions=sum(a['status']=='rejected' for a in row['attempts']),
            stopped=row['checkedStop'],externally_truncated=not row['checkedStop'])
        if endpoint!=row['endpoint']:
            raise ValueError('RECORDED_SEQUENTIAL_SELECTED_ENDPOINT_CHANGED')
    return True
