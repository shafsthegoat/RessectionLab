"""Generated bridge-consumer controls, NOT evidence-registry admission.

Only resolve_critical_evidence is replaced with an explicitly fictional typed
resolved result. Geometry, planners, bridge requests, stale-binding checks and
real-source admission remain unchanged. No human review is invented. These
controls cannot establish source authenticity or a compliant learned result.
"""
from pathlib import Path
from types import MappingProxyType
import json
import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest, freeze_json, immutable_array
from resectionlab import critical_evidence as critical
from resectionlab.desktop_bridge import BridgeRuntime, BridgeSession, BridgeError
from resectionlab.geometry import GENERIC_TOOLS
from resectionlab.imaging import save_case


def generated_case(blocked):
    shape=(25,25,25)
    brain=np.zeros(shape,bool);brain[2:23,2:23,2:23]=True
    grid=np.indices(shape)
    target=sum((grid[a]-(17,12,12)[a])**2 for a in range(3))<=9
    return CaseData('generated-critical-consumer-control',brain.astype(np.float32),{'target':target},np.eye(4),
        (SourceRef('generated','synthetic:critical-consumer-control',provenance='simulated'),),brain_mask=brain,
        metadata={'software_control_only':True,'generated_critical_blocked':blocked,
                  'human_review_performed':False,'patient_admission':False})


def generated_constraints(case):
    assert case.metadata.get('software_control_only') is True
    mask=np.zeros(case.mri.shape,bool)
    if case.metadata['generated_critical_blocked']:
        mask|=case.brain_mask
    mask=immutable_array(mask)
    coverage=immutable_array(case.brain_mask)
    receipt={'schema':'critical-constraints-v1','records':{'generated-resolver-control':{
        'structure':'vessels','exclusion_reason':None,'mask_hash':array_digest(mask),
        'coverage_hash':array_digest(coverage),'scope':'generated_resolver_injection_only',
        'human_review_performed':False,'real_registry_admission':False}},
        'missing':['motor','language'],'objective_structures':[],
        'time_scope':'generated_software_control_no_patient_clock',
        'coverage_meaning':'generated domain, not acquired vessel completeness',
        'trajectory_coverage':'not_computed','clinical_clearance':False}
    return critical.CriticalConstraints(
        MappingProxyType({'motor':None,'language':None,'vessels':mask}),
        MappingProxyType({'motor':None,'language':None,'vessels':coverage}),freeze_json(receipt),case.mri.shape)


class Client:
    def __init__(self,path):
        self.events=[];self.runtime=BridgeRuntime(path,self.events.append);self.serial=0
    def call(self,op,args,error=False):
        self.serial+=1;key=str(self.serial)
        self.runtime.submit({'id':key,'op':op,'args':args})
        assert self.runtime.wait_idle(15)
        result=[row for row in self.events if row['id']==key][-1]
        assert result['event']==('error' if error else 'result'),result
        return result['error'] if error else result['result']


@pytest.mark.parametrize('operation',['generateRoutes','generateNativeRoutes'])
def test_actual_bridge_routes_consume_generated_resolved_masks_and_reject_stale_binding(tmp_path,monkeypatch,operation):
    monkeypatch.setattr(critical,'resolve_critical_evidence',generated_constraints)
    clear,blocked=generated_case(False),generated_case(True)
    client=Client(tmp_path/'bridge')
    try:
        records=[];opened=[]
        for name,case in [('clear',clear),('blocked',blocked)]:
            path=save_case(case,tmp_path/f'{name}.ressectionlab')
            loaded=client.call('loadCase',{'path':str(path)});opened.append(loaded)
            assert loaded['criticalEvidence']['records']['generated-resolver-control']['human_review_performed'] is False
            args={'caseHash':loaded['caseHash']}
            if operation=='generateRoutes':
                args.update(toolIds=[GENERIC_TOOLS[0].tool_id],config={'targetsPerCompartment':1,'maxWindows':1})
            records.append(client.call(operation,args))
        before,after=records
        assert before['candidates'] and all(row['geometry']['feasible'] for row in before['candidates'])
        assert after['candidates'] and all(not row['geometry']['feasible'] for row in after['candidates'])
        assert len(before['candidates'])==len(after['candidates'])
        for left,right in zip(before['candidates'],after['candidates']):
            assert all(left[key]==right[key] for key in ('entry_mm','target_mm','tool','window'))
            assert any(item['reason']=='FORBIDDEN_COLLISION' for item in right['geometry']['failures'])
        assert before['planning_model_hash']!=after['planning_model_hash']
        assert all(row['structure_contact_volume_mm3']['vessels']>0 for row in after['candidates'])
        assert all(row['category']=='rejected' and row['simulated_removed_target_volume_mm3'] is None for row in after['candidates'])
        assert all(row['clinical_deficit_probability'] is None for row in after['candidates'])
        old_entry=client.runtime.session.cases[opened[0]['caseHash']]
        config=BridgeSession._route_config(old_entry,before['candidates'][0]['route_id'])
        assert config['criticalEvidenceHash']==generated_constraints(clear).fingerprint
        with pytest.raises(BridgeError,match='Critical evidence differs'):
            BridgeSession._run_options(config,blocked)
        stale=client.call('inspectRefinement',{'caseHash':opened[1]['caseHash'],
            'routeId':before['candidates'][0]['route_id']},error=True)
        assert stale['code']=='ROUTE_UNAVAILABLE'
        assert clear.critical_evidence==blocked.critical_evidence=={}
        (tmp_path/'consumer-result.json').write_text(json.dumps({'operation':operation,
            'scope':'generated_resolver_injection_only_no_real_admission',
            'unchanged_requested_geometry':True,'before':before,'after':after,
            'stale_route_error':stale['code'],'real_registry_admission':False},indent=2)+'\n')
    finally:
        client.runtime.close()


def test_generated_consumer_seam_does_not_admit_real_registry_or_learning(tmp_path,monkeypatch):
    case=generated_case(True)
    with pytest.raises(ValueError,match='cannot be simulated'):
        critical.CriticalStructureEvidence(evidence_id='generated',structure='vessels',
            mask=case.brain_mask,annotation_coverage=case.brain_mask,affine_ras_mm=case.affine,
            case_id=case.case_id,reference_image_hash=array_digest(case.mri),reference_source_id='generated',
            reference_source_sha256='0'*64,source=SourceRef('generated','synthetic:control',
                sha256='0'*64,license='generated software control',provenance='simulated'),
            source_binding={},derivation={},lineage={})
    monkeypatch.setattr(critical,'resolve_critical_evidence',generated_constraints)
    from resectionlab.native_spatial_task import native_spatial_task_from_case
    with pytest.raises(ValueError,match='CRITICAL_EVIDENCE_UNSUPPORTED'):
        native_spatial_task_from_case(case,access=None,tools=())
    client=Client(tmp_path/'bridge')
    try:
        refusal=client.call('nativeTraining',{'caseHash':case.semantic_hash},error=True)
        assert refusal['code']=='RECORDED_EXPERIENCE_REQUIRED'
    finally:
        client.runtime.close()
