"""Explicit fixed-four TRAIN capability; not a PatientTrainSession/checkpoint path.

The existing source/QC validation still runs in full before any inventory.
Its historical search-only receipt authenticates the public world only. The
separately root-released pilot authorizes exactly one ridge solve and top8
inference; it does not relabel that receipt as neural training admission.
"""
from dataclasses import dataclass
from resectionlab.core import freeze_json,semantic_digest
from resectionlab.native_spatial_task import NativeSpatialCase
from hybrid_proposal import GeneratedSparseTask,VERSION

SUBJECTS=('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045')
PILOT_VERSION='fixed-four-TRAIN-public-ridge-proposal-v1'

@dataclass(frozen=True)
class PilotAdmission:
    record: object
    fingerprint: str

    def require_source(self,case):
        if (semantic_digest(self.record)!=self.fingerprint or type(case) is not NativeSpatialCase
            or self.record['version']!=PILOT_VERSION or self.record['subject'] not in SUBJECTS
            or self.record['role']!='TRAIN' or case.track!='annotation_assisted'
            or case.source_hash!=self.record['source_hash']
            or case.reference_hash!=self.record['reference_hash']
            or case._native_config.fingerprint!=self.record['native_config_hash']):
            raise ValueError('Fixed public TRAIN pilot source changed')
        case.assert_intact()


def admit_source(case,*,cohort_bytes,source_binding,qc_receipt,protocol,pilot_declaration):
    # No task construction or observation is used by this canonical validator.
    from resectionlab.patient_planning_admission import validate_patient_planning_source
    validate_patient_planning_source(case,cohort_bytes=cohort_bytes,source_binding=source_binding,
        qc_receipt=qc_receipt,protocol=protocol)
    if (set(pilot_declaration)!={'version','released_sha256','source_world_hashes','proposal_config','post_exposure_condition'}
        or pilot_declaration['version']!=PILOT_VERSION
        or type(pilot_declaration['released_sha256']) is not str
        or len(pilot_declaration['released_sha256'])!=64
        or any(c not in '0123456789abcdef' for c in pilot_declaration['released_sha256'])
        or tuple(pilot_declaration['source_world_hashes'])!=SUBJECTS
        or source_binding['subject'] not in SUBJECTS or protocol['role']!='TRAIN'
        or source_binding['evidence_domain']!='acquired_patient'
        or protocol['max_steps']!=24 or protocol['max_optimizer_updates']!=0
        or protocol['max_policy_forwards']!=0 or protocol['initialization']!='public_world_search_only'
        or case.post_exposure is None or case.post_exposure.record['version']!=pilot_declaration['post_exposure_condition']
        or case.proposal_mode!='nominal_cavity_v1' or case.proposal_config.obstruction_opening
        or case.proposal_config.max_candidates!=120 or len(case.proposal_config.offsets_source_voxels)!=13
        or not case.proposal_config.tool_footprint_opening or case.proposal_config.intermediate_opening_mm!=1.
        or semantic_digest(case.proposal_config.to_record())!=semantic_digest(pilot_declaration['proposal_config'])
        or case.source_hash!=pilot_declaration['source_world_hashes'][source_binding['subject']]):
        raise ValueError('Exact fixed-four TRAIN ridge pilot declaration required')
    record=freeze_json({'version':PILOT_VERSION,'subject':source_binding['subject'],'role':'TRAIN',
        'source_hash':case.source_hash,'reference_hash':case.reference_hash,'native_config_hash':case._native_config.fingerprint,
        'source_validation_protocol_hash':semantic_digest(protocol),'source_binding_hash':semantic_digest(source_binding),
        'qc_hash':semantic_digest(qc_receipt),'pilot_declaration':pilot_declaration,
        'feature_version':VERSION,'private_reference_used':False,'max_steps':24,
        'scope':'one fixed TRAIN ridge priority solve; exact verified top8; no SELECT/EVAL or neural updates'})
    return PilotAdmission(record,semantic_digest(record))


class AdmittedSparseTask(GeneratedSparseTask):
    def __init__(self,case,*,admission,mode,ranker=None,max_steps=24,cancelled=None,costs=None):
        if type(admission) is not PilotAdmission or max_steps!=24:
            raise ValueError('Explicit fixed TRAIN pilot admission required')
        self.admission=admission
        super().__init__(case,mode=mode,ranker=ranker,max_steps=max_steps,cancelled=cancelled,costs=costs)

    def _require_source(self,case):
        self.admission.require_source(case)

    def _contract_record(self):
        return {**super()._contract_record(),'fixed_TRAIN_pilot_admission':self.admission.fingerprint}

    def fresh(self):
        self._assert_frozen()
        return type(self)(self.case,admission=self.admission,mode=self._selection,ranker=self._ranker,
            max_steps=self.max_steps,cancelled=self._cancelled,costs=self.costs)
