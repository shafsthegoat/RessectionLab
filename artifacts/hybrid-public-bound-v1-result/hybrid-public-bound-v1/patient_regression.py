"""Inference-only successor of the original fixed-four PilotAdmission.

Historical qualification authenticates the same world. This capability is not
an outside-fit development-transfer admission and cannot fit, train or evaluate
SELECT/EVAL. The only new authority is the root-released frozen-policy regression.
"""
from dataclasses import dataclass
import json
import numpy as np
from resectionlab.core import freeze_json,semantic_digest
from resectionlab.native_spatial_task import NativeSpatialCase
from hybrid_proposal import GeneratedSparseTask,RidgePriority,VERSION as FEATURE_VERSION
from fallback_policy import NoPositiveFallbackMixin
from batch_contract import VERSION,RIDGE,PROPOSALS,EXPOSURE,FIT_SUBJECTS as SUBJECTS,TRAIN_RESULT,read_bytes,need,digest

def load_frozen_ridge():
    """Called once only inside the released owned worker; never fit or torch.load."""
    raw=read_bytes(RIDGE['path'],RIDGE['sha256'],limit=256*1024)
    obj=json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite_ridge')))
    need(set(obj)=={'feature_version','data_hash','mean','scale','weights','intercept'},'exact_ridge_JSON_schema')
    need(obj['feature_version']==RIDGE['fingerprint'] and obj['data_hash']==RIDGE['data_hash'],'original_fit_lineage')
    for key in ('mean','scale','weights'):
        need(type(obj[key]) is list and len(obj[key])==681 and all(type(x) in (int,float) for x in obj[key]),'finite_numeric_ridge_vector')
    need(type(obj['intercept']) in (int,float),'numeric_ridge_intercept')
    ranker=RidgePriority(*(np.asarray(obj[k],np.float64) for k in ('mean','scale','weights')),float(obj['intercept']))
    need(ranker.fingerprint==RIDGE['fingerprint'],'exact_frozen_ridge_parameters')
    return ranker


@dataclass(frozen=True)
class PilotAdmission:
    record: object
    fingerprint: str
    def require_source(self,case):
        need(semantic_digest(self.record)==self.fingerprint and type(case) is NativeSpatialCase
             and self.record['version']==VERSION and self.record['subject'] in SUBJECTS
             and self.record['role']=='TRAIN' and case.track=='annotation_assisted'
             and case.source_hash==self.record['source_hash'] and case.reference_hash==self.record['reference_hash']
             and case._native_config.fingerprint==self.record['native_config_hash'],'exact_fixed_four_regression_world')
        case.assert_intact()


def admit_source(case,*,cohort_bytes,source_binding,qc_receipt,protocol,pilot_declaration):
    from resectionlab.patient_planning_admission import validate_patient_planning_source
    validate_patient_planning_source(case,cohort_bytes=cohort_bytes,source_binding=source_binding,qc_receipt=qc_receipt,protocol=protocol)
    d=pilot_declaration
    need(set(d)=={'version','released_sha256','source_world_hashes','proposal_config','post_exposure_condition',
                 'fits','optimizer_updates','neural_forwards','frozen_ridge','training_result'}
         and d['version']==VERSION and d['fits']==d['optimizer_updates']==d['neural_forwards']==0
         and all(type(d[k]) is int for k in ('fits','optimizer_updates','neural_forwards'))
         and d['frozen_ridge']==RIDGE and d['training_result']==TRAIN_RESULT
         and tuple(d['source_world_hashes'])==tuple(SUBJECTS)
         and source_binding['subject'] in SUBJECTS and protocol['role']=='TRAIN'
         and source_binding['evidence_domain']=='acquired_patient'
         and protocol['max_steps']==24 and protocol['max_optimizer_updates']==protocol['max_policy_forwards']==0
         and protocol['initialization']=='public_world_search_only' and 'development_transfer' not in protocol
         and case.post_exposure is not None and case.post_exposure.record['version']==d['post_exposure_condition']==EXPOSURE
         and case.proposal_mode=='nominal_cavity_v1' and not case.proposal_config.obstruction_opening
         and case.proposal_config.max_candidates==120 and len(case.proposal_config.offsets_source_voxels)==13
         and case.proposal_config.tool_footprint_opening and case.proposal_config.intermediate_opening_mm==1.
         and semantic_digest(case.proposal_config.to_record())==semantic_digest(d['proposal_config'])==semantic_digest(PROPOSALS)
         and case.source_hash==d['source_world_hashes'][source_binding['subject']],'exact_fixed_four_zero_fit_declaration')
    digest(d['released_sha256'])
    record=freeze_json({'version':VERSION,'subject':source_binding['subject'],'role':'TRAIN',
        'source_hash':case.source_hash,'reference_hash':case.reference_hash,'native_config_hash':case._native_config.fingerprint,
        'source_validation_protocol_hash':semantic_digest(protocol),'source_binding_hash':semantic_digest(source_binding),
        'qc_hash':semantic_digest(qc_receipt),'pilot_declaration':d,'feature_version':FEATURE_VERSION,
        'private_reference_used':False,'max_steps':24,'scope':'frozen fitted-cohort inference regression; zero fit/update/neural; SELECT/EVAL closed'})
    return PilotAdmission(record,semantic_digest(record))


class AdmittedRegressionTask(NoPositiveFallbackMixin,GeneratedSparseTask):
    def __init__(self,case,*,admission,ranker,max_steps=24,cancelled=None,costs=None,mode='ridge_top8'):
        need(type(admission) is PilotAdmission and type(max_steps) is int and max_steps==24,'explicit_fixed_four_capability')
        need(mode=='ridge_top8' and type(ranker) is RidgePriority and ranker.fingerprint==RIDGE['fingerprint'],'frozen_ranker_only')
        self.admission=admission
        super().__init__(case,mode=mode,ranker=ranker,max_steps=max_steps,cancelled=cancelled,costs=costs)
    def _require_source(self,case):self.admission.require_source(case)
    def _contract_record(self):return {**super()._contract_record(),'fixed_TRAIN_regression_admission':self.admission.fingerprint}
    def fresh(self):
        self._assert_frozen()
        return type(self)(self.case,admission=self.admission,ranker=self._ranker,
            max_steps=self.max_steps,cancelled=self._cancelled,costs=self.costs)
