"""Source-aware admission for new planning; inspection/replay confer no permission."""
from __future__ import annotations
from .core import semantic_digest

SCHEMA = 'resectionlab.planning-admission.v1'
REQUIRED = 'PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING'
_PATIENT_KEYS = frozenset(('patient', 'sourcePatient', 'patientId', 'sourceManifest',
    'sourceCollectionHash', 'source_patient', 'source_manifest', 'source_collection_hash'))


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
    evidence = ('generated-engineering' if generated else 'patient-or-unverified',
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
    if not callable(getattr(scene, 'require_cache_source', None)):
        return 'patient-or-unverified'
    scene.require_cache_source()
    validate_source_binding(scene)
    source = scene.source
    if _acquired_markers(source):
        return 'patient-or-unverified'
    evidence = getattr(scene, '_planning_source_evidence', None)
    if evidence is not None:
        return evidence[0]
    if source.get('caseHash') is not None:
        return 'patient-or-unverified'
    if (source.get('evidenceClass') == 'generated-engineering' or
            source.get('kind') in ('generated-control', 'engineering-control') or
            source.get('kind') == 'public-planning-observation' and
            source.get('track') == 'annotation-assisted-inactive-access-v1'):
        return 'generated-engineering'
    return 'patient-or-unverified'


def planning_admission(scene, protection=None):
    from .surgical_procedure_policy import PROPERTY, procedure_admission
    from .surgical_source_inspection import inspection_only, inspection_blockers
    classification = source_classification(scene)
    if inspection_only(scene):
        return {'schema': SCHEMA, 'available': False, 'scope': 'source-inspection-only',
            'sourceClass': classification, 'blockers': inspection_blockers(scene),
            'policyHash': None, 'clinicalValidation': False}
    if PROPERTY in scene.properties:
        procedure = procedure_admission(scene, protection)
        return {'schema': SCHEMA, 'available': procedure['available'], 'scope': 'procedure-aware',
            'sourceClass': classification, 'blockers': procedure['blockers'],
            'policyHash': procedure.get('policyHash'), 'clinicalValidation': False}
    if classification == 'generated-engineering':
        return {'schema': SCHEMA, 'available': True, 'scope': 'legacy-engineering',
            'sourceClass': classification, 'blockers': [], 'policyHash': None,
            'clinicalValidation': False}
    return {'schema': SCHEMA, 'available': False, 'scope': 'unsupported-patient-planning',
        'sourceClass': classification, 'policyHash': None, 'clinicalValidation': False,
        'blockers': [{'code': REQUIRED, 'message':
            'Patient-derived or unverified anatomy requires the source-bound procedure policy; legacy engineering planning cannot authorize this source.'}]}


def require_planning_admission(scene_or_engine, protection=None):
    scene = getattr(scene_or_engine, 'scene', scene_or_engine)
    if protection is None:
        protection = getattr(scene_or_engine, 'protection', None)
    admission = planning_admission(scene, protection)
    if not admission['available']:
        block = admission['blockers'][0]
        raise ValueError(block['code'] + ': ' + block['message'])
    return admission


def require_case_planning_admission(case):
    from .surgical_patient_role import patient_role_evidence
    adapter = case.metadata.get('surgical_adapter') or {}
    if (_acquired_markers({'adapter': dict(adapter)}) or
            patient_role_evidence(case).get('status') != 'generated_control'):
        raise ValueError(REQUIRED + ': native learning requires explicit generated source evidence; patient anatomy cannot select a legacy task by omitting its policy')
    return {'schema': SCHEMA, 'available': True, 'scope': 'legacy-engineering'}
