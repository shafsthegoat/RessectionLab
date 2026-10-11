"""Fabricated annotation-style sources; no acquired inputs or QC claim."""
import copy,importlib.util,sys
from pathlib import Path
import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(Path(__file__).parent))
import test_post_exposure_learning as generated
from test_source_admission import staged
from resectionlab import patient_planning_admission as canonical
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.core import semantic_digest
from patient_sparse import admit_source,AdmittedSparseTask,PILOT_VERSION,SUBJECTS


def prepared(tmp_path,monkeypatch,subject='ReMIND-002'):
    module=staged();monkeypatch.setattr(canonical,'validate_patient_planning_source',module.validate_patient_planning_source,raising=False)
    inputs=generated.fixture(tmp_path,subject);p=generated.protocol()
    case,binding,qc,protocol=generated.build(tmp_path/'source',inputs,p,search_only=True)
    args=dict(cohort_bytes=inputs['cohort_bytes'],source_binding=binding,qc_receipt=qc,protocol=protocol)
    declaration={'version':PILOT_VERSION,'released_sha256':'a'*64,
        'source_world_hashes':{s:case.source_hash for s in SUBJECTS},
        'proposal_config':case.proposal_config.to_record(),'post_exposure_condition':case.post_exposure.record['version']}
    return case,args,declaration


def test_capability_before_preview_and_sparse_actual_constructor(tmp_path,monkeypatch):
    case,args,declaration=prepared(tmp_path,monkeypatch)
    original=NativeResectionEngine.preview_stroke
    def forbidden(*a,**k):raise AssertionError('eager preview during admission')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    token=admit_source(case,**args,pilot_declaration=declaration)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',original)
    task=AdmittedSparseTask(case,admission=token,mode='sampled_target_top8')
    assert task.costs.counts['native_previews']<=8 and task.max_steps==24
    task.planning_clone().observation();task.fresh()
    task.step('STOP');assert task.independent_geometry_check().feasible


@pytest.mark.parametrize('change',['subject','source','role','private','updates','horizon','config','release'])
def test_mismatched_capability_fails_before_inventory(tmp_path,monkeypatch,change):
    case,args,d=prepared(tmp_path,monkeypatch)
    if change=='subject':d['source_world_hashes']['ReMIND-037']=d['source_world_hashes'].pop('ReMIND-002')
    if change=='source':d['source_world_hashes']['ReMIND-002']='sha256:'+'f'*64
    if change=='role':args['protocol']['role']='SELECT'
    if change=='private':args['protocol']['private_reference_used']=True
    if change=='updates':args['protocol']['max_optimizer_updates']=1
    if change=='horizon':args['protocol']['max_steps']=6
    if change=='config':d['proposal_config']['max_candidates']=96
    if change=='release':d['released_sha256']='not-a-release'
    def forbidden(*a,**k):raise AssertionError('native preview before refusal')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    with pytest.raises(ValueError):admit_source(case,**args,pilot_declaration=d)
