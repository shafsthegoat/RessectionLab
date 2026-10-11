"""Direct frozen-ridge capability for exactly one untrained TRAIN050 source."""
from dataclasses import dataclass
import json
import numpy as np
from resectionlab.core import freeze_json,semantic_digest
from resectionlab.native_spatial_task import NativeSpatialCase
from hybrid_proposal import GeneratedSparseTask,RidgePriority,VERSION as FEATURE_VERSION
from batch_contract import VERSION,RIDGE,PROPOSALS,EXPOSURE,read_bytes,need,SUBJECT,TRAIN_RESULT


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
class TransferAdmission:
    record: object
    fingerprint: str
    def require_source(self,case):
        need(semantic_digest(self.record)==self.fingerprint and type(case) is NativeSpatialCase
             and self.record['version']==VERSION and self.record['subject']==SUBJECT
             and self.record['role']=='TRAIN' and case.track=='annotation_assisted'
             and case.source_hash==self.record['source_hash'] and case.reference_hash==self.record['reference_hash']
             and case._native_config.fingerprint==self.record['native_config_hash'],'exact_admitted_050_public_world')
        case.assert_intact()


def admit_source(case,*,cohort_bytes,source_binding,qc_receipt,protocol,release_sha256):
    from resectionlab.patient_planning_admission import validate_patient_planning_source
    validate_patient_planning_source(case,cohort_bytes=cohort_bytes,source_binding=source_binding,qc_receipt=qc_receipt,protocol=protocol)
    d=protocol['development_transfer']
    need(d['version']==VERSION and d['frozen_ridge_sha256']==RIDGE['sha256'] and d['frozen_ridge_fingerprint']==RIDGE['fingerprint'] and d['training_result_sha256']==TRAIN_RESULT['sha256']
         and protocol['runtime_release_sha256']==release_sha256 and case.post_exposure is not None
         and case.post_exposure.record['version']==EXPOSURE and case.proposal_mode=='nominal_cavity_v1'
         and not case.proposal_config.obstruction_opening and semantic_digest(case.proposal_config.to_record())==semantic_digest(PROPOSALS),
         'fixed_ridge_world_and_source_contract')
    record=freeze_json({'version':VERSION,'subject':SUBJECT,'role':'TRAIN','source_hash':case.source_hash,
        'reference_hash':case.reference_hash,'native_config_hash':case._native_config.fingerprint,
        'source_binding_hash':semantic_digest(source_binding),'qc_hash':semantic_digest(qc_receipt),
        'protocol_hash':semantic_digest(protocol),'feature_version':FEATURE_VERSION,'development_transfer':d,
        'runtime_release_sha256':release_sha256,'private_reference_used':False})
    return TransferAdmission(record,semantic_digest(record))


class AdmittedTransferTask(GeneratedSparseTask):
    def __init__(self,case,*,admission,mode,ranker=None,max_steps=24,cancelled=None,costs=None):
        need(type(admission) is TransferAdmission and type(max_steps) is int and max_steps==24,'explicit_050_capability')
        need(mode in ('ridge_top8','exhaustive') and ((mode=='ridge_top8' and type(ranker) is RidgePriority
             and ranker.fingerprint==RIDGE['fingerprint']) or (mode=='exhaustive' and ranker is None)),'frozen_ridge_or_lean_control_only')
        self.admission=admission
        super().__init__(case,mode=mode,ranker=ranker,max_steps=max_steps,cancelled=cancelled,costs=costs)
    def _require_source(self,case):self.admission.require_source(case)
    def _contract_record(self):return {**super()._contract_record(),'TRAIN_development_transfer':self.admission.fingerprint}
    def fresh(self):
        self._assert_frozen()
        return type(self)(self.case,admission=self.admission,mode=self._selection,ranker=self._ranker,
            max_steps=self.max_steps,cancelled=self._cancelled,costs=self.costs)
