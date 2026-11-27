"""Source-labelled admission for new planning; inspection/replay confer no permission.

Owner decision 2026-11-27: planning, search, sequential tasks and the RL
environment operate on any 3D reconstruction, including acquired patient
anatomy. The source class is reported with every admission and is never a
blocker. Structural requirements (a declared target, procedure memberships and
budgets when a procedure policy is present) are unchanged and still name what
is missing. Provenance labels stay: a generated source is labelled
generated-engineering; anything else is patient-or-unverified.
"""
from __future__ import annotations
from .core import semantic_digest

SCHEMA = 'resectionlab.planning-admission.v1'
# Retired refusal code. Kept so historical receipts and guard-proof readers
# that compare against it still import; no admission path raises it now.
REQUIRED = 'PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING'
GENERATED = 'generated-engineering'
ACQUIRED = 'patient-or-unverified'
_PATIENT_KEYS = frozenset(('patient', 'sourcePatient', 'patientId', 'sourceManifest',
    'sourceCollectionHash', 'source_patient', 'source_manifest', 'source_collection_hash',
    'patient_id', 'source_patient_id', 'sourcePatientId'))


def _acquired_markers(source):
    adapter = source.get('adapter') or {}
    return (bool(_PATIENT_KEYS.intersection(source)) or
        source.get('procedurePolicySourceClass') == 'acquired-or-unknown' or
        isinstance(adapter, dict) and (bool(_PATIENT_KEYS.intersection(adapter)) or
            str(adapter.get('type', '')).startswith(('native-patient', 'nifti', 'mesh'))))


def bind_case_source(scene, case):
    """Bind actual CaseData provenance without changing historic scene IDs."""
    from .surgical_patient_role import patient_role_evidence
    generated = patient_role_evidence(case).get('status') == 'generated_control'
    evidence = (GENERATED if generated else ACQUIRED,
        case.semantic_hash, semantic_digest([ref.to_dict() for ref in case.source_refs]))
    scene._planning_source_evidence = evidence
    scene._planning_source_proof = (evidence, scene.identity)


def validate_source_binding(scene):
    evidence = getattr(scene, '_planning_source_evidence', None)
    proof = getattr(scene, '_planning_source_proof', None)
    if evidence is None and proof is None:
        return
    if (type(evidence) is not tuple or len(evidence) != 3 or type(proof) is not tuple or
            len(proof) != 2 or proof[0] is not evidence or proof[1] != scene.identity or
            evidence[1] != scene.source.get('caseHash')):
        raise ValueError('PROCEDURE_SOURCE_PROVENANCE_CHANGED')


def propagate_source_binding(source, target):
    validate_source_binding(source)
    evidence = getattr(source, '_planning_source_evidence', None)
    if evidence is not None and target.source.get('caseHash') == evidence[1]:
        target._planning_source_evidence = evidence
        target._planning_source_proof = (evidence, target.identity)
    return target


def source_classification(scene):
    """Provenance label of a scene's source. A label, never an admission gate."""
    if not callable(getattr(scene, 'require_cache_source', None)):
        return ACQUIRED
    scene.require_cache_source()
    validate_source_binding(scene)
    source = scene.source
    if _acquired_markers(source):
        return ACQUIRED
    evidence = getattr(scene, '_planning_source_evidence', None)
    if evidence is not None:
        return evidence[0]
    if source.get('caseHash') is not None:
        return ACQUIRED
    if (source.get('evidenceClass') == GENERATED or
            source.get('kind') in ('generated-control', 'engineering-control') or
            isinstance(source.get('adapter'), dict) and
            source['adapter'].get('type') == 'generated-engineering-control' or
            source.get('kind') == 'public-planning-observation' and
            source.get('track') == 'annotation-assisted-inactive-access-v1'):
        return GENERATED
    return ACQUIRED


def planning_admission(scene, protection=None):
    from .surgical_procedure_policy import PROPERTY, procedure_admission
    from .surgical_source_inspection import inspection_only, inspection_blockers
    from .surgical_patient_role import source_role_label
    classification = source_classification(scene)
    role = source_role_label(scene, classification)
    if inspection_only(scene):
        # The engine executes no native action on a source-inspection export.
        # This mirrors that execution contract; it is not a source-class gate.
        return {'schema': SCHEMA, 'available': False, 'scope': 'source-inspection-only',
            'sourceClass': classification, 'sourceRole': role, 'blockers': inspection_blockers(scene),
            'policyHash': None, 'clinicalValidation': False}
    if PROPERTY in scene.properties:
        procedure = procedure_admission(scene, protection)
        return {'schema': SCHEMA, 'available': procedure['available'], 'scope': 'procedure-aware',
            'sourceClass': classification, 'sourceRole': role, 'blockers': procedure['blockers'],
            'policyHash': procedure.get('policyHash'), 'clinicalValidation': False}
    # Without a procedure policy the legacy engineering contract applies to any
    # source. Its harm model is the engine's guards; unsupplied anatomy stays
    # UNKNOWN in each planner's report, never zero risk.
    return {'schema': SCHEMA, 'available': True, 'scope': 'legacy-engineering',
        'sourceClass': classification, 'sourceRole': role, 'blockers': [], 'policyHash': None,
        'clinicalValidation': False}


def require_planning_admission(scene_or_engine, protection=None):
    scene = getattr(scene_or_engine, 'scene', scene_or_engine)
    if protection is None:
        protection = getattr(scene_or_engine, 'protection', None)
    admission = planning_admission(scene, protection)
    if not admission['available']:
        block = admission['blockers'][0]
        raise ValueError(block['code'] + ': ' + block['message'])
    return admission


def case_source_classification(case):
    """CaseData provenance label; acquired markers or observed references win."""
    from .surgical_patient_role import patient_role_evidence
    adapter = case.metadata.get('surgical_adapter') or {}
    if (_acquired_markers({'adapter': dict(adapter)}) or
            patient_role_evidence(case).get('status') != 'generated_control'):
        return ACQUIRED
    return GENERATED


def require_case_planning_admission(case):
    """Legacy native-learning admission for any case, labelled with its source."""
    return {'schema': SCHEMA, 'available': True, 'scope': 'legacy-engineering',
        'sourceClass': case_source_classification(case), 'clinicalValidation': False}
