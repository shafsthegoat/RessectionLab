"""Read-only source/task readiness. A diagnostic never grants planner authority."""
from __future__ import annotations

import numpy as np

from .core import semantic_digest
from .surgical_task_admission import admission, source_evidence, verify_source, validate_admission

SCHEMA = 'resectionlab.surgical-source-preflight.v1'


def source_preflight(engine, *, native_collection=None, verify_replay=False, check=lambda: None):
    check(); engine.scene.require_cache_source()
    binding = (engine.scene.identity, engine.state_id, semantic_digest(engine.actions))
    checks = []
    def add(identity, status, description):
        checks.append({'id': identity, 'status': status, 'description': description})
    evidence = None; verification = None
    try:
        evidence = source_evidence(engine.scene)
        add('source-role', 'passed', 'Existing source declarations qualify DEVELOPMENT use; anatomical/task qualification remains separate.')
    except ValueError as error:
        if type(error) is not ValueError: raise
        add('source-role', 'refused', str(error))
    if evidence is not None and native_collection is not None:
        try:
            verification = verify_source(engine.scene, native_collection=native_collection, check=check)
            add('source-bytes', 'passed', 'Native original bytes, array hashes, frames and material ownership match the source adapter.')
        except ValueError as error:
            if type(error) is not ValueError: raise
            add('source-bytes', 'refused', str(error))
    else:
        add('source-bytes', 'not-checked', 'Native original/array byte verification requires the matching source collection and an eligible source role.')
    if verify_replay and verification is not None:
        from .surgical_engine import SurgicalEngine
        try:
            SurgicalEngine.restore(engine.scene, engine.snapshot(), check=check, use_cache=False)
            add('native-replay', 'passed', 'The complete recorded native checkpoint replays exactly.')
        except ValueError as error:
            if type(error) is not ValueError: raise
            add('native-replay', 'refused', str(error))
    else: add('native-replay', 'not-checked', 'No additional complete native replay was requested and source-byte qualified here.')
    modern = engine.state.get('toolMechanicsRevision') == 2 and engine.state.get('physicalClockRevision') == 1
    add('model-versions', 'passed' if modern else 'required', 'This task requires tool geometry revision2 and shared physical clock revision1.')
    inactive = not engine.state['holds'] and not any(s['active'] or s['inserted'] or s['grip'] for s in engine.state['slots'].values())
    add('withdrawn-baseline', 'passed' if inactive else 'required', 'Both instruments must be inactive, released and withdrawn at the planning checkpoint.')
    removed = len(engine.state['removed'])
    add('executed-exposure', 'passed' if removed else 'required',
        f'{removed} source cells removed in the native state. A task still requires a clear entry linked to actual local removal and retained target.')
    undeformed = not any(np.linalg.norm(v) > 0. for v in engine.state.get('deformation', {}).values())
    add('static-baseline', 'passed' if undeformed else 'required', 'The current bounded geometry task requires an undeformed native baseline.')
    bound = admission(engine); admitted = False
    if bound is None: add('declared-task', 'required', 'Declare and verify bounded entry, target, direction, permitted inputs/actions and research acceptance after completing exposure.')
    else:
        try:
            validate_admission(engine, bound); admitted = True
            add('declared-task', 'passed', 'The existing source-bound task admission validates at this exact checkpoint.')
        except ValueError as error:
            if type(error) is not ValueError: raise
            add('declared-task', 'refused', str(error))
    unreviewed = sorted(k for k, row in (evidence or {}).get('anatomy', {}).items() if row['reviewStatus'] != 'reviewed')
    accepted = admitted and bound['task']['allowUnreviewedEvidence']
    add('evidence-review', ('accepted-for-research' if accepted else 'requires-research-opt-in') if unreviewed
        else 'not-assessed' if evidence is None else 'passed',
        'Unreviewed anatomy: '+', '.join(unreviewed)+'. Explicit research acceptance does not constitute anatomical review.' if unreviewed
        else 'Review status follows supplied source evidence; missing anatomy remains unknown.')
    check(); engine.scene.require_cache_source()
    if binding != (engine.scene.identity, engine.state_id, semantic_digest(engine.actions)):
        raise ValueError('SOURCE_PREFLIGHT: native checkpoint changed during inspection')
    if evidence is not None and evidence != source_evidence(engine.scene):
        raise ValueError('SOURCE_PREFLIGHT: source declarations changed during inspection')
    value = {'schema': SCHEMA, 'sourceSceneId': binding[0], 'startStateId': binding[1], 'actionsHash': binding[2],
        'prefixActions': len(engine.actions), 'planningAdmitted': admitted, 'sourceEvidence': evidence,
        'sourceVerification': verification, 'checks': checks,
        'scope': 'Read-only source/task readiness. This diagnostic creates no exposure, task admission, candidate or permission to execute. Source annotations/estimates are not clinical outcomes; patient-specific uncertainty and missing anatomy remain unqualified.'}
    value['preflightSeal'] = semantic_digest(value)
    return value
